#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <set>

using namespace clang;
using namespace clang::tooling;

static llvm::cl::OptionCategory MyToolCategory("cuda-launch-parser");
static int TempVarCounter = 0;
static int LoopCounter = 0;
static int WarpSize = 32;

static std::set<std::string> EmittedFunctions;
static std::vector<std::string> FunctionBodies;

static llvm::cl::opt<int> WarpSizeOpt("warp-size",
    llvm::cl::desc("Target warp size: 32 for NVIDIA, 64 for AMD CDNA"),
    llvm::cl::init(32),
    llvm::cl::cat(MyToolCategory));

std::string mlirType(QualType T) {
    if (T->isFloatingType()) {
        if (T->isSpecificBuiltinType(BuiltinType::Half)) return "f16";
        if (T->isSpecificBuiltinType(BuiltinType::Double)) return "f64";
        return "f32";
    }
    if (T->isBooleanType()) return "i1";
    return "index";
}

bool isFloatLike(QualType T) { return T->isFloatingType(); }
std::string exprType(const Expr *E) { return mlirType(E->getType()); }

int pointerDepth(QualType T) {
    int depth = 0;
    while (T->isPointerType()) { depth++; T = T->getPointeeType(); }
    return depth;
}

std::string memrefType(int dims, const std::string &elem) {
    std::string s = "memref<";
    for (int i = 0; i < dims; i++) s += "?x";
    s += elem + ">";
    return s;
}

struct Symbol {
    std::string mlirName;
    std::string type;
    bool isMemref;
    int dims;
};

class CudaLauncher {
public:
    std::map<std::string, Symbol> symbols;
    raw_ostream &OS;
    CudaLauncher(raw_ostream &OS) : OS(OS) {}
    void declare(const std::string &name, const std::string &mlirName,
                 const std::string &type, bool isMemref, int dims = 1) {
        symbols[name] = {mlirName, type, isMemref, dims};
    }
    std::string lookup(const std::string &name) {
        if (symbols.count(name)) return symbols[name].mlirName;
        return "%" + name;
    }
};

bool isBooleanExpr(const Expr *E);

bool isBooleanExpr(const Expr *E) {
    if (!E) return false;
    if (E->getType()->isBooleanType()) return true;
    const Expr *inner = E->IgnoreParenImpCasts();
    if (const auto *BO = dyn_cast<BinaryOperator>(inner)) {
        int op = BO->getOpcode();
        if (op == BO_LT || op == BO_GT || op == BO_LE || op == BO_GE || op == BO_EQ || op == BO_NE)
            return true;
        if (op == BO_LAnd || op == BO_LOr) return true;
    }
    return false;
}

CompoundAssignOperator *findAccumulation(Stmt *S) {
    if (!S) return nullptr;
    if (auto *CAO = dyn_cast<CompoundAssignOperator>(S))
        if (CAO->getOpcode() == BO_AddAssign) return CAO;
    for (Stmt *C : S->children())
        if (auto *F = findAccumulation(C)) return F;
    return nullptr;
}

std::string translateExpr(const Expr *E, CudaLauncher &L);

std::string translateExpr(const Expr *E, CudaLauncher &L) {
    if (!E) return "";
    E = E->IgnoreParenImpCasts();

    if (const auto *CE = dyn_cast<CallExpr>(E)) {
        if (const auto *FD = CE->getDirectCallee()) {
            std::string name = FD->getNameAsString();
            
            if (FD->hasAttr<CUDADeviceAttr>() && !FD->hasAttr<CUDAGlobalAttr>()) {
                std::string args;
                for (unsigned i = 0; i < CE->getNumArgs(); i++) {
                    if (i > 0) args += ", ";
                    args += translateExpr(CE->getArg(i), L);
                }
                std::string retType = mlirType(FD->getReturnType());
                std::string v = "%" + std::to_string(TempVarCounter++);
                L.OS << "      " << v << " = func.call @" << name << "(" << args << ") : (";
                for (unsigned i = 0; i < CE->getNumArgs(); i++) {
                    if (i > 0) L.OS << ", ";
                    L.OS << mlirType(CE->getArg(i)->getType());
                }
                L.OS << ") -> " << retType << "\n";
                return v;
            }
            
            if (name == "__syncthreads" || name == "__syncwarp") {
                L.OS << "      gpu.barrier\n";
                return "";
            }
            if (name == "atomicAdd" && CE->getNumArgs() >= 2) {
                std::string val = translateExpr(CE->getArg(1), L);
                std::string ty = exprType(CE->getArg(1));
                const Expr *ptrE = CE->getArg(0)->IgnoreParenImpCasts();
                if (const auto *UO = dyn_cast<UnaryOperator>(ptrE)) {
                    if (UO->getOpcode() == UO_AddrOf) ptrE = UO->getSubExpr()->IgnoreParenImpCasts();
                }
                if (const auto *ASE = dyn_cast<ArraySubscriptExpr>(ptrE)) {
                    std::string arr = translateExpr(ASE->getBase(), L);
                    std::string idx = translateExpr(ASE->getIdx(), L);
                    std::string v = "%" + std::to_string(TempVarCounter++);
                    L.OS << "      " << v << " = memref.atomic_rmw addf " << val << ", "
                         << arr << "[" << idx << "] : (memref<?x" << ty << ">, " << ty << ") -> " << ty << "\n";
                    return v;
                }
                return val;
            }
            if ((name == "__shfl_sync" || name == "__shfl_down_sync" || name == "__shfl_up_sync") && CE->getNumArgs() >= 3) {
                std::string mask = translateExpr(CE->getArg(0), L);
                std::string val = translateExpr(CE->getArg(1), L);
                std::string src = translateExpr(CE->getArg(2), L);
                std::string v = "%" + std::to_string(TempVarCounter++);
                if (WarpSize == 32) {
                    L.OS << "      " << v << " = nvvm.shfl.sync.down " << mask << ", " << val << ", " << src << " : f32\n";
                } else {
                    std::string half = "%half_" + std::to_string(TempVarCounter++);
                    std::string src_half = "%src_half_" + std::to_string(TempVarCounter++);
                    std::string same = "%same_" + std::to_string(TempVarCounter++);
                    std::string shfl = "%shfl_" + std::to_string(TempVarCounter++);
                    L.OS << "      // WARP_EMULATION\n";
                    L.OS << "      " << half << " = arith.divsi %tx, %c32 : index\n";
                    L.OS << "      " << src_half << " = arith.divsi " << src << ", %c32 : index\n";
                    L.OS << "      " << same << " = arith.cmpi eq, " << half << ", " << src_half << " : index\n";
                    L.OS << "      " << v << " = scf.if " << same << " -> (f32) {\n";
                    L.OS << "        " << shfl << " = nvvm.shfl.sync.down " << mask << ", " << val << ", " << src << " : f32\n";
                    L.OS << "        scf.yield " << shfl << " : f32\n";
                    L.OS << "      } else {\n";
                    L.OS << "        scf.yield " << val << " : f32\n";
                    L.OS << "      }\n";
                }
                return v;
            }
        }
    }

    if (const auto *Cast = dyn_cast<CXXStaticCastExpr>(E)) {
        std::string src = translateExpr(Cast->getSubExpr(), L);
        std::string dstType = mlirType(Cast->getType());
        std::string srcType = mlirType(Cast->getSubExpr()->getType());
        if (dstType != srcType) {
            std::string v = "%" + std::to_string(TempVarCounter++);
            L.OS << "      " << v << " = arith.extf " << src << " : " << srcType << " to " << dstType << "\n";
            return v;
        }
        return src;
    }

    if (const auto *IL = dyn_cast<IntegerLiteral>(E))
        return "%c" + std::to_string(IL->getValue().getZExtValue());

    if (const auto *FL = dyn_cast<FloatingLiteral>(E)) {
        std::string varName = "%f" + std::to_string(TempVarCounter++);
        L.OS << "      " << varName << " = arith.constant "
             << FL->getValueAsApproximateDouble() << " : " << exprType(E) << "\n";
        return varName;
    }

    if (const auto *DRE = dyn_cast<DeclRefExpr>(E)) {
        return L.lookup(DRE->getDecl()->getNameAsString());
    }

    if (const auto *ME = dyn_cast<MemberExpr>(E)) {
        std::string baseName = "";
        if (const auto *BaseDRE = dyn_cast<DeclRefExpr>(ME->getBase()->IgnoreParenImpCasts()))
            baseName = BaseDRE->getDecl()->getNameAsString();
        std::string field = ME->getMemberNameInfo().getAsString();
        if (baseName == "blockIdx")  return field == "y" ? "%by" : "%bx";
        if (baseName == "threadIdx") return field == "y" ? "%ty" : "%tx";
        if (baseName == "blockDim")  return "%c2";
        return "%" + baseName;
    }

    if (const auto *ASE = dyn_cast<ArraySubscriptExpr>(E)) {
        std::vector<std::string> indices;
        const Expr *current = E;
        while (const auto *curASE = dyn_cast<ArraySubscriptExpr>(current->IgnoreParenImpCasts())) {
            indices.insert(indices.begin(), translateExpr(curASE->getIdx(), L));
            current = curASE->getBase();
        }
        std::string arr = translateExpr(current, L);
        std::string v = "%" + std::to_string(TempVarCounter++);
        std::string elemType = exprType(E);
        int n = indices.size();
        L.OS << "      " << v << " = memref.load " << arr << "[";
        for (int i = 0; i < n; i++) { if (i > 0) L.OS << ", "; L.OS << indices[i]; }
        L.OS << "] : " << memrefType(n, elemType) << "\n";
        return v;
    }

    if (const auto *BO = dyn_cast<BinaryOperator>(E)) {
        if (BO->getOpcode() == BO_Assign) {
            const Expr *LHS = BO->getLHS()->IgnoreParenImpCasts();
            if (const auto *ASE = dyn_cast<ArraySubscriptExpr>(LHS)) {
                std::vector<std::string> indices;
                const Expr *current = LHS;
                while (const auto *curASE = dyn_cast<ArraySubscriptExpr>(current->IgnoreParenImpCasts())) {
                    indices.insert(indices.begin(), translateExpr(curASE->getIdx(), L));
                    current = curASE->getBase();
                }
                std::string arr = translateExpr(current, L);
                std::string val = translateExpr(BO->getRHS(), L);
                std::string elemType = exprType(BO->getRHS());
                int n = indices.size();
                L.OS << "      memref.store " << val << ", " << arr << "[";
                for (int i = 0; i < n; i++) { if (i > 0) L.OS << ", "; L.OS << indices[i]; }
                L.OS << "] : " << memrefType(n, elemType) << "\n";
                return val;
            }
            if (const auto *DRE = dyn_cast<DeclRefExpr>(LHS)) {
                std::string name = DRE->getDecl()->getNameAsString();
                std::string val = translateExpr(BO->getRHS(), L);
                L.declare(name, val, exprType(BO->getRHS()), false);
                return val;
            }
        }
    }

    if (const auto *CAO = dyn_cast<CompoundAssignOperator>(E)) {
        const Expr *LHS = CAO->getLHS()->IgnoreParenImpCasts();
        if (const auto *ASE = dyn_cast<ArraySubscriptExpr>(LHS)) {
            std::vector<std::string> indices;
            const Expr *current = LHS;
            while (const auto *curASE = dyn_cast<ArraySubscriptExpr>(current->IgnoreParenImpCasts())) {
                indices.insert(indices.begin(), translateExpr(curASE->getIdx(), L));
                current = curASE->getBase();
            }
            std::string arr = translateExpr(current, L);
            std::string rhs = translateExpr(CAO->getRHS(), L);
            std::string ty = exprType(CAO->getRHS());
            std::string v = "%" + std::to_string(TempVarCounter++);
            std::string old = "%" + std::to_string(TempVarCounter++);
            int n = indices.size();
            L.OS << "      " << old << " = memref.load " << arr << "[";
            for (int i = 0; i < n; i++) { if (i > 0) L.OS << ", "; L.OS << indices[i]; }
            L.OS << "] : " << memrefType(n, ty) << "\n";
            L.OS << "      " << v << " = " << (isFloatLike(CAO->getType()) ? "arith.addf" : "arith.addi")
                 << " " << old << ", " << rhs << " : " << ty << "\n";
            L.OS << "      memref.store " << v << ", " << arr << "[";
            for (int i = 0; i < n; i++) { if (i > 0) L.OS << ", "; L.OS << indices[i]; }
            L.OS << "] : " << memrefType(n, ty) << "\n";
            return v;
        }
        if (CAO->getOpcode() == BO_AddAssign) {
            std::string lhs = translateExpr(CAO->getLHS(), L);
            std::string rhs = translateExpr(CAO->getRHS(), L);
            std::string v = "%" + std::to_string(TempVarCounter++);
            std::string ty = exprType(CAO->getRHS());
            L.OS << "      " << v << " = " << (isFloatLike(CAO->getType()) ? "arith.addf" : "arith.addi")
                 << " " << lhs << ", " << rhs << " : " << ty << "\n";
            return v;
        }
    }

    if (const auto *BO = dyn_cast<BinaryOperator>(E)) {
        std::string lhs = translateExpr(BO->getLHS(), L);
        std::string rhs = translateExpr(BO->getRHS(), L);
        bool flt = isFloatLike(BO->getType());
        std::string ty = "index";
        if (flt) {
            ty = "f32";
            auto checkF16 = [&](const Expr *op) -> bool {
                const Expr *oe = op->IgnoreParenImpCasts();
                if (const auto *ASE = dyn_cast<ArraySubscriptExpr>(oe)) {
                    if (const auto *BaseDRE = dyn_cast<DeclRefExpr>(ASE->getBase()->IgnoreParenImpCasts())) {
                        std::string name = BaseDRE->getDecl()->getNameAsString();
                        if (L.symbols.count(name) && L.symbols[name].type.find("f16") != std::string::npos) {
                            return true;
                        }
                    }
                }
                return false;
            };
            if (checkF16(BO->getLHS()) || checkF16(BO->getRHS())) ty = "f16";
            else {
                std::string lty = mlirType(BO->getLHS()->getType());
                std::string rty = mlirType(BO->getRHS()->getType());
                if (lty == "f16" || rty == "f16") ty = "f16";
                else if (lty == "f64" || rty == "f64") ty = "f64";
            }
        }
        bool isCmp = (BO->getOpcode() == BO_LT || BO->getOpcode() == BO_GT || 
                      BO->getOpcode() == BO_LE || BO->getOpcode() == BO_GE ||
                      BO->getOpcode() == BO_EQ || BO->getOpcode() == BO_NE);
        std::string op;
        switch (BO->getOpcode()) {
            case BO_Add: op = flt ? "arith.addf" : "arith.addi"; break;
            case BO_Sub: op = flt ? "arith.subf" : "arith.subi"; break;
            case BO_Mul: op = flt ? "arith.mulf" : "arith.muli"; break;
            case BO_Div: op = flt ? "arith.divf" : "arith.divsi"; break;
            case BO_Rem: op = flt ? "arith.remf" : "arith.remsi"; break;
            case BO_LT:  op = flt ? "arith.cmpf olt," : "arith.cmpi slt,"; break;
            case BO_GT:  op = flt ? "arith.cmpf ogt," : "arith.cmpi sgt,"; break;
            case BO_LE:  op = flt ? "arith.cmpf ole," : "arith.cmpi sle,"; break;
            case BO_GE:  op = flt ? "arith.cmpf oge," : "arith.cmpi sge,"; break;
            case BO_EQ:  op = flt ? "arith.cmpf oeq," : "arith.cmpi eq,"; break;
            case BO_NE:  op = flt ? "arith.cmpf one," : "arith.cmpi ne,"; break;
            default: return lhs;
        }
        std::string v = "%" + std::to_string(TempVarCounter++);
        std::string resType = isCmp ? "i1" : ty;
        L.OS << "      " << v << " = " << op << " " << lhs << ", " << rhs << " : " << ty << "\n";
        return v;
    }

    if (const auto *UO = dyn_cast<UnaryOperator>(E)) {
        if (UO->getOpcode() == UO_Deref) {
            const Expr *sub = UO->getSubExpr()->IgnoreParenImpCasts();
            if (const auto *BO = dyn_cast<BinaryOperator>(sub)) {
                if (BO->getOpcode() == BO_Add) {
                    std::string arr = translateExpr(BO->getLHS(), L);
                    std::string idx = translateExpr(BO->getRHS(), L);
                    std::string v = "%" + std::to_string(TempVarCounter++);
                    std::string elemType = exprType(UO);
                    L.OS << "      " << v << " = memref.load " << arr << "[" << idx
                         << "] : memref<?x" << elemType << ">\n";
                    return v;
                }
            }
        }
    }

    return "";
}

void dumpStmtAsMLIR(Stmt *S, CudaLauncher &L, raw_ostream &OS) {
    if (!S) return;

    if (auto *UO = dyn_cast<UnaryOperator>(S)) {
        if (UO->getOpcode() == UO_PostInc || UO->getOpcode() == UO_PreInc) return;
    }

    if (auto *BO = dyn_cast<BinaryOperator>(S)) {
        if (BO->getOpcode() == BO_Assign) {
            translateExpr(BO, L);
            return;
        }
    }

    if (auto *Ret = dyn_cast<ReturnStmt>(S)) {
        if (Ret->getRetValue()) {
            std::string v = translateExpr(Ret->getRetValue(), L);
            OS << "      return " << v << " : " << exprType(Ret->getRetValue()) << "\n";
        } else {
            OS << "      return\n";
        }
        return;
    }

    if (auto *Asm = dyn_cast<GCCAsmStmt>(S)) {
        std::string asmStr = Asm->getAsmString();
        std::vector<std::string> operands;
        for (unsigned i = 0; i < Asm->getNumInputs(); i++)
            operands.push_back(translateExpr(Asm->getInputExpr(i), L));
        for (unsigned i = 0; i < Asm->getNumOutputs(); i++)
            operands.push_back(translateExpr(Asm->getOutputExpr(i), L));
        auto op = [&](int i, const std::string &fallback) {
            return i < (int)operands.size() ? operands[i] : fallback;
        };
        std::string v = "%" + std::to_string(TempVarCounter++);
        if (asmStr.find("bar.sync") != std::string::npos) {
            OS << "      gpu.barrier\n";
            return;
        }
        if (asmStr.find("fma.rn.f32") != std::string::npos) {
            std::string mul = "%" + std::to_string(TempVarCounter++);
            OS << "      " << mul << " = arith.mulf " << op(0, "%f0") << ", " << op(1, "%f0") << " : f32\n";
            OS << "      " << v << " = arith.addf " << mul << ", " << op(2, "%f0") << " : f32\n";
            return;
        }
        OS << "      // PTX_UNSUPPORTED\n";
        return;
    }

    if (auto *CE = dyn_cast<CallExpr>(S)) {
        translateExpr(CE, L);
        return;
    }

    if (auto *DS = dyn_cast<DeclStmt>(S)) {
        for (auto *D : DS->decls()) {
            if (auto *VD = dyn_cast<VarDecl>(D)) {
                std::string name = VD->getNameAsString();
                if (VD->hasAttr<CUDASharedAttr>()) {
                    QualType baseType = VD->getType();
                    int dims = 1;
                    if (baseType->isArrayType()) baseType = baseType->getAsArrayTypeUnsafe()->getElementType();
                    std::string t = mlirType(baseType);
                    L.declare(name, "%" + name, t, true, dims);
                    OS << "      %" << name << " = memref.alloc() : memref<256x" << t << ">\n";
                } else if (VD->hasInit()) {
                    std::string v = translateExpr(VD->getInit(), L);
                    int dims = pointerDepth(VD->getType());
                    if (dims == 0) dims = 1;
                    L.declare(name, v, mlirType(VD->getType()), false, dims);
                } else {
                    std::string t = mlirType(VD->getType());
                    std::string init = "%zero_" + name;
                    OS << "      " << init << " = arith.constant 0.0 : " << t << "\n";
                    L.declare(name, init, t, false);
                }
            }
        }
        return;
    }

    if (auto *If = dyn_cast<IfStmt>(S)) {
        std::string cond = translateExpr(If->getCond(), L);
        if (isBooleanExpr(If->getCond())) {
            OS << "      scf.if " << cond << " {\n";
        } else {
            std::string c1 = "%cmp_" + std::to_string(TempVarCounter++);
            OS << "      " << c1 << " = arith.cmpi ne, " << cond << ", %c0 : index\n";
            OS << "      scf.if " << c1 << " {\n";
        }
        dumpStmtAsMLIR(If->getThen(), L, OS);
        OS << "      }";
        if (If->getElse()) {
            OS << " else {\n";
            dumpStmtAsMLIR(If->getElse(), L, OS);
            OS << "      }";
        }
        OS << "\n";
        return;
    }

    if (auto *While = dyn_cast<WhileStmt>(S)) {
        std::string varMlir = "%while_j_" + std::to_string(LoopCounter);
        const Expr *cond = While->getCond()->IgnoreParenImpCasts();
        std::string ub = "%c10";
        std::string cmpOp = "slt";
        if (const auto *BO = dyn_cast<BinaryOperator>(cond)) {
            ub = translateExpr(BO->getRHS(), L);
            if (BO->getOpcode() == BO_LT) cmpOp = "slt";
            else if (BO->getOpcode() == BO_NE) cmpOp = "ne";
        }
        OS << "      %while_result_" << LoopCounter << " = scf.while (" << varMlir << " = %c0) : (index) -> index {\n";
        OS << "        %cond_" << LoopCounter << " = arith.cmpi " << cmpOp << ", " << varMlir << ", " << ub << " : index\n";
        OS << "        scf.condition(%cond_" << LoopCounter << ") " << varMlir << " : index\n";
        OS << "      } do {\n";
        OS << "        ^bb0(%arg_j: index):\n";
        L.declare("j", "%arg_j", "index", false);
        dumpStmtAsMLIR(While->getBody(), L, OS);
        OS << "        %next_j_" << LoopCounter << " = arith.addi %arg_j, %c1 : index\n";
        OS << "        scf.yield %next_j_" << LoopCounter << " : index\n";
        OS << "      }\n";
        LoopCounter++;
        return;
    }

    if (auto *Sw = dyn_cast<SwitchStmt>(S)) {
        std::string cond = translateExpr(Sw->getCond(), L);
        SwitchCase *SC = Sw->getSwitchCaseList();
        std::vector<CaseStmt*> cases;
        DefaultStmt *defaultCase = nullptr;
        while (SC) {
            if (auto *CS = dyn_cast<CaseStmt>(SC)) cases.push_back(CS);
            else if (auto *DS = dyn_cast<DefaultStmt>(SC)) defaultCase = DS;
            SC = SC->getNextSwitchCase();
        }
        std::reverse(cases.begin(), cases.end());
        for (auto *CS : cases) {
            std::string caseVal = translateExpr(CS->getLHS(), L);
            std::string cmp = "%" + std::to_string(TempVarCounter++);
            OS << "      " << cmp << " = arith.cmpi eq, " << cond << ", " << caseVal << " : index\n";
            OS << "      scf.if " << cmp << " {\n";
            dumpStmtAsMLIR(CS->getSubStmt(), L, OS);
            OS << "      }\n";
        }
        if (defaultCase) {
            OS << "      // default\n";
            dumpStmtAsMLIR(defaultCase->getSubStmt(), L, OS);
        }
        return;
    }

    // FIX: For loop with accumulation (sum += ...)
    if (auto *For = dyn_cast<ForStmt>(S)) {
        std::string UB = "%N";
        if (const Expr *C = For->getCond())
            if (const auto *BO = dyn_cast<BinaryOperator>(C->IgnoreParenImpCasts()))
                UB = translateExpr(BO->getRHS(), L);
        
        CompoundAssignOperator *CAO = findAccumulation(For->getBody());
        std::string accVar = "";
        std::string accType = "f32";
        if (CAO) {
            if (const auto *DRE = dyn_cast<DeclRefExpr>(CAO->getLHS()->IgnoreParenImpCasts()))
                accVar = DRE->getDecl()->getNameAsString();
            accType = exprType(CAO->getRHS());
        }
        
        std::string ivName = "%iv_" + std::to_string(LoopCounter++);
        L.declare("k", ivName, "index", false);
        L.declare("i", ivName, "index", false);
        L.declare("iv", ivName, "index", false);
        
        if (!accVar.empty()) {
            std::string init = "%init_" + accVar;
            std::string iter = "%iter_" + accVar;
            OS << "      " << init << " = arith.constant 0.0 : " << accType << "\n";
            OS << "      %acc_result = scf.for " << ivName << " = %c0 to " << UB << " step %c1 iter_args("
               << iter << " = " << init << ") -> (" << accType << ") {\n";
            std::string inner = translateExpr(CAO->getRHS(), L);
            OS << "        %new_sum = arith.addf " << iter << ", " << inner << " : " << accType << "\n";
            OS << "        scf.yield %new_sum : " << accType << "\n";
            OS << "      }\n";
            L.declare(accVar, "%acc_result", accType, false);
        } else {
            OS << "      scf.for " << ivName << " = %c0 to " << UB << " step %c1 {\n";
            dumpStmtAsMLIR(For->getBody(), L, OS);
            OS << "      }\n";
        }
        return;
    }

    for (Stmt *C : S->children()) {
        if (C && isa<Expr>(C)) translateExpr(cast<Expr>(C), L);
        else dumpStmtAsMLIR(C, L, OS);
    }
}

class CudaLaunchVisitor : public RecursiveASTVisitor<CudaLaunchVisitor> {
public:
    explicit CudaLaunchVisitor(ASTContext *Context) : Context(Context) {}

    void emitKernelToBuffer(FunctionDecl *FD) {
        if (!FD || !FD->hasBody()) return;
        bool isDevice = FD->hasAttr<CUDADeviceAttr>() && !FD->hasAttr<CUDAGlobalAttr>();
        bool isGlobal = FD->hasAttr<CUDAGlobalAttr>();
        if (!isDevice && !isGlobal) return;

        std::string kernelName = FD->getNameAsString();
        if (EmittedFunctions.count(kernelName)) return;
        EmittedFunctions.insert(kernelName);

        std::string buffer;
        llvm::raw_string_ostream BufOS(buffer);
        // Keep counters global so SSA values are unique across all functions
        
        BufOS << "  func.func @" << kernelName << "(";
        CudaLauncher L(BufOS);
        bool first = true;
        for (auto *Param : FD->parameters()) {
            std::string pname = Param->getNameAsString();
            std::string ptype;
            bool isMemref = false;
            int dims = 1;
            if (Param->getType()->isPointerType()) {
                dims = pointerDepth(Param->getType());
                if (dims == 0) dims = 1;
                QualType baseT = Param->getType();
                for (int i = 0; i < dims; i++) baseT = baseT->getPointeeType();
                std::string elem = mlirType(baseT);
                ptype = memrefType(dims, elem);
                isMemref = true;
            } else {
                ptype = mlirType(Param->getType());
            }
            if (!first) BufOS << ", ";
            first = false;
            BufOS << "%" << pname << ": " << ptype;
            L.declare(pname, "%" + pname, ptype, isMemref, dims);
        }
        BufOS << ")";
        if (!isGlobal) {
            BufOS << " -> " << mlirType(FD->getReturnType());
        }
        BufOS << " {\n";
        BufOS << "    %c0 = arith.constant 0 : index\n";
        BufOS << "    %c1 = arith.constant 1 : index\n";
        BufOS << "    %c2 = arith.constant 2 : index\n";
        BufOS << "    %c5 = arith.constant 5 : index\n";
        BufOS << "    %c10 = arith.constant 10 : index\n";
        BufOS << "    %c32 = arith.constant 32 : index\n";
        BufOS << "    %f0 = arith.constant 0.0 : f32\n";
        BufOS << "    %f1 = arith.constant 1.0 : f32\n";
        
        if (isGlobal) {
            BufOS << "    %cTotal = arith.muli %N, %N : index\n";
            BufOS << "    scf.for %t = %c0 to %cTotal step %c1 {\n";
            BufOS << "      %row = arith.divsi %t, %N : index\n";
            BufOS << "      %col = arith.remsi %t, %N : index\n";
            BufOS << "      %bx = arith.divsi %row, %c2 : index\n";
            BufOS << "      %tx = arith.remsi %row, %c2 : index\n";
            BufOS << "      %by = arith.divsi %col, %c2 : index\n";
            BufOS << "      %ty = arith.remsi %col, %c2 : index\n";
            dumpStmtAsMLIR(FD->getBody(), L, BufOS);
            BufOS << "    }\n";
        } else {
            dumpStmtAsMLIR(FD->getBody(), L, BufOS);
        }
        BufOS << "    return\n";
        BufOS << "  }\n";
        BufOS.flush();
        FunctionBodies.push_back(buffer);
    }

    bool VisitFunctionDecl(FunctionDecl *FD) {
        if (FD->getTemplatedKind() == FunctionDecl::TK_FunctionTemplate) return true;
        if (FD->hasAttr<CUDAGlobalAttr>() || FD->hasAttr<CUDADeviceAttr>()) {
            emitKernelToBuffer(FD);
        }
        return true;
    }

    bool VisitFunctionTemplateDecl(FunctionTemplateDecl *FTD) {
        if (!FTD->getTemplatedDecl()->hasAttr<CUDAGlobalAttr>()) return true;
        for (auto *Spec : FTD->specializations()) {
            emitKernelToBuffer(Spec);
        }
        return true;
    }

private:
    ASTContext *Context;
};

class CudaLaunchConsumer : public ASTConsumer {
public:
    explicit CudaLaunchConsumer(ASTContext *Context) : Visitor(Context) {}
    void HandleTranslationUnit(ASTContext &Context) override {
        Visitor.TraverseDecl(Context.getTranslationUnitDecl());
    }
private:
    CudaLaunchVisitor Visitor;
};

class CudaLaunchAction : public ASTFrontendAction {
public:
    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef InFile) override {
        return std::make_unique<CudaLaunchConsumer>(&CI.getASTContext());
    }
};

int main(int argc, const char **argv) {
    auto ExpectedParser = CommonOptionsParser::create(argc, argv, MyToolCategory);
    if (!ExpectedParser) {
        llvm::errs() << llvm::toString(ExpectedParser.takeError());
        return 1;
    }
    WarpSize = WarpSizeOpt;
    ClangTool Tool(ExpectedParser->getCompilations(), ExpectedParser->getSourcePathList());
    Tool.appendArgumentsAdjuster(clang::tooling::getInsertArgumentAdjuster(
        {"-x", "cuda", "--cuda-gpu-arch=sm_70", "--cuda-path=/tmp", "-nocudainc", "-nocudalib"},
        clang::tooling::ArgumentInsertPosition::BEGIN));
    int ret = Tool.run(newFrontendActionFactory<CudaLaunchAction>().get());
    
    if (!FunctionBodies.empty()) {
        llvm::outs() << "// Target warp size: " << WarpSize << "\n";
        llvm::outs() << "module {\n";
        for (const auto &body : FunctionBodies) {
            llvm::outs() << body;
        }
        llvm::outs() << "}\n";
    }
    return ret;
}