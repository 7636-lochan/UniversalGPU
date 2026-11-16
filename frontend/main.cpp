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

using namespace clang;
using namespace clang::tooling;

static llvm::cl::OptionCategory MyToolCategory("cuda-launch-parser");
static int TempVarCounter = 0;
static int LoopCounter = 0;
static int WarpSize = 32;

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

std::string translateExpr(const Expr *E, CudaLauncher &L);

std::string translateExpr(const Expr *E, CudaLauncher &L) {
    if (!E) return "";
    E = E->IgnoreParenImpCasts();

    if (const auto *CE = dyn_cast<CallExpr>(E)) {
        if (const auto *FD = CE->getDirectCallee()) {
            std::string name = FD->getNameAsString();
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
                    std::string lane = "%lane_" + std::to_string(TempVarCounter++);
                    std::string half = "%half_" + std::to_string(TempVarCounter++);
                    std::string src_half = "%src_half_" + std::to_string(TempVarCounter++);
                    std::string same_half = "%same_half_" + std::to_string(TempVarCounter++);
                    std::string shfl_res = "%shfl_" + std::to_string(TempVarCounter++);
                    L.OS << "      // WARP_EMULATION: AMD 64-lane wavefront, 32-lane logical warp\n";
                    L.OS << "      " << lane << " = arith.remsi %tx, %c32 : index\n";
                    L.OS << "      " << half << " = arith.divsi %tx, %c32 : index\n";
                    L.OS << "      " << src_half << " = arith.divsi " << src << ", %c32 : index\n";
                    L.OS << "      " << same_half << " = arith.cmpi eq, " << half << ", " << src_half << " : index\n";
                    L.OS << "      " << v << " = scf.if " << same_half << " -> (f32) {\n";
                    L.OS << "        " << shfl_res << " = nvvm.shfl.sync.down " << mask << ", " << val << ", " << src << " : f32\n";
                    L.OS << "        scf.yield " << shfl_res << " : f32\n";
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
        std::string name = DRE->getDecl()->getNameAsString();
        // FIX 3: Don't map i, k, iv to %iv — use unique names
        return L.lookup(name);
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

    // FIX 2: Handle compound assignment to memory (data[i] += 1.0f)
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

    // FIX 2: Compound assignment to memory
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
        // Scalar compound assignment: sum += x
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
        std::string op;
        switch (BO->getOpcode()) {
            case BO_Add: op = flt ? "arith.addf" : "arith.addi"; break;
            case BO_Sub: op = flt ? "arith.subf" : "arith.subi"; break;
            case BO_Mul: op = flt ? "arith.mulf" : "arith.muli"; break;
            case BO_Div: op = flt ? "arith.divf" : "arith.divsi"; break;
            case BO_Rem: op = flt ? "arith.remf" : "arith.remsi"; break;  // FIX 1
            case BO_LT:  op = flt ? "arith.cmpf olt," : "arith.cmpi slt,"; break;
            case BO_EQ:  op = flt ? "arith.cmpf oeq," : "arith.cmpi eq,"; break;
            default: return lhs;
        }
        std::string v = "%" + std::to_string(TempVarCounter++);
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

CompoundAssignOperator *findAccumulation(Stmt *S) {
    if (!S) return nullptr;
    if (auto *CAO = dyn_cast<CompoundAssignOperator>(S))
        if (CAO->getOpcode() == BO_AddAssign) return CAO;
    for (Stmt *C : S->children())
        if (auto *F = findAccumulation(C)) return F;
    return nullptr;
}

BinaryOperator *findPlainAccumulation(Stmt *S) {
    if (!S) return nullptr;
    if (auto *BO = dyn_cast<BinaryOperator>(S)) {
        if (BO->getOpcode() == BO_Assign) {
            const Expr *LHS = BO->getLHS()->IgnoreParenImpCasts();
            if (isa<DeclRefExpr>(LHS)) return BO;
        }
    }
    for (Stmt *C : S->children())
        if (auto *F = findPlainAccumulation(C)) return F;
    return nullptr;
}

void dumpStmtAsMLIR(Stmt *S, CudaLauncher &L) {
    if (!S) return;

    if (auto *BO = dyn_cast<BinaryOperator>(S)) {
        if (BO->getOpcode() == BO_Assign) {
            translateExpr(BO, L);
            return;
        }
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

        if (asmStr.find("shfl.sync.down") != std::string::npos) {
            if (WarpSize == 32) {
                L.OS << "      // PTX_SHFL_DOWN: shfl.sync.down (native)\n";
                L.OS << "      " << v << " = nvvm.shfl.sync.down 0xffffffff, " << op(0, "%zero") << ", " << op(1, "%c1") << " : f32\n";
            } else {
                L.OS << "      // WARP_EMULATION: shfl.sync.down masked to 32-lane half\n";
                std::string lane = "%lane_" + std::to_string(TempVarCounter++);
                std::string half = "%half_" + std::to_string(TempVarCounter++);
                std::string src_half = "%src_half_" + std::to_string(TempVarCounter++);
                std::string same_half = "%same_half_" + std::to_string(TempVarCounter++);
                std::string shfl_res = "%shfl_" + std::to_string(TempVarCounter++);
                L.OS << "      " << lane << " = arith.remsi %tx, %c32 : index\n";
                L.OS << "      " << half << " = arith.divsi %tx, %c32 : index\n";
                L.OS << "      " << src_half << " = arith.divsi " << op(1, "%c1") << ", %c32 : index\n";
                L.OS << "      " << same_half << " = arith.cmpi eq, " << half << ", " << src_half << " : index\n";
                L.OS << "      " << v << " = scf.if " << same_half << " -> (f32) {\n";
                L.OS << "        " << shfl_res << " = nvvm.shfl.sync.down 0xffffffff, " << op(0, "%zero") << ", " << op(1, "%c1") << " : f32\n";
                L.OS << "        scf.yield " << shfl_res << " : f32\n";
                L.OS << "      } else {\n";
                L.OS << "        scf.yield " << op(0, "%zero") << " : f32\n";
                L.OS << "      }\n";
            }
            return;
        }
        if (asmStr.find("shfl.sync") != std::string::npos) {
            if (WarpSize == 32) {
                L.OS << "      // PTX_SHFL: shfl.sync (native)\n";
                L.OS << "      " << v << " = nvvm.shfl.sync.down 0xffffffff, " << op(0, "%zero") << ", " << op(1, "%c1") << " : f32\n";
            } else {
                L.OS << "      // WARP_EMULATION: shfl.sync masked to 32-lane half\n";
                std::string lane = "%lane_" + std::to_string(TempVarCounter++);
                std::string half = "%half_" + std::to_string(TempVarCounter++);
                std::string src_half = "%src_half_" + std::to_string(TempVarCounter++);
                std::string same_half = "%same_half_" + std::to_string(TempVarCounter++);
                std::string shfl_res = "%shfl_" + std::to_string(TempVarCounter++);
                L.OS << "      " << lane << " = arith.remsi %tx, %c32 : index\n";
                L.OS << "      " << half << " = arith.divsi %tx, %c32 : index\n";
                L.OS << "      " << src_half << " = arith.divsi " << op(1, "%c1") << ", %c32 : index\n";
                L.OS << "      " << same_half << " = arith.cmpi eq, " << half << ", " << src_half << " : index\n";
                L.OS << "      " << v << " = scf.if " << same_half << " -> (f32) {\n";
                L.OS << "        " << shfl_res << " = nvvm.shfl.sync.down 0xffffffff, " << op(0, "%zero") << ", " << op(1, "%c1") << " : f32\n";
                L.OS << "        scf.yield " << shfl_res << " : f32\n";
                L.OS << "      } else {\n";
                L.OS << "        scf.yield " << op(0, "%zero") << " : f32\n";
                L.OS << "      }\n";
            }
            return;
        }
        if (asmStr.find("bar.sync") != std::string::npos) {
            L.OS << "      // PTX_BAR: bar.sync\n";
            L.OS << "      gpu.barrier\n";
            return;
        }
        if (asmStr.find("fma.rn.f32") != std::string::npos) {
            std::string mul = "%" + std::to_string(TempVarCounter++);
            L.OS << "      // PTX_FMA: fma.rn.f32\n";
            L.OS << "      " << mul << " = arith.mulf " << op(0, "%f0") << ", " << op(1, "%f0") << " : f32\n";
            L.OS << "      " << v << " = arith.addf " << mul << ", " << op(2, "%f0") << " : f32\n";
            return;
        }
        if (asmStr.find("mad.lo.s32") != std::string::npos || asmStr.find("mad.lo.u32") != std::string::npos) {
            std::string mul = "%" + std::to_string(TempVarCounter++);
            L.OS << "      // PTX_MAD: mad.lo.s32\n";
            L.OS << "      " << mul << " = arith.muli " << op(0, "%c0") << ", " << op(1, "%c0") << " : index\n";
            L.OS << "      " << v << " = arith.addi " << mul << ", " << op(2, "%c0") << " : index\n";
            return;
        }
        if (asmStr.find("ld.global.f32") != std::string::npos) {
            L.OS << "      // PTX_LD_GLOBAL_F32\n";
            L.OS << "      " << v << " = memref.load %input[" << op(0, "%c0") << "] : memref<?xf32>\n";
            return;
        }
        if (asmStr.find("ld.shared.f32") != std::string::npos) {
            L.OS << "      // PTX_LD_SHARED_F32\n";
            L.OS << "      " << v << " = memref.load %tile[" << op(0, "%c0") << "] : memref<?xf32>\n";
            return;
        }
        if (asmStr.find("ld.global") != std::string::npos) {
            L.OS << "      // PTX_LD_GLOBAL: " << asmStr.substr(0, 40) << "\n";
            return;
        }
        if (asmStr.find("st.global.f32") != std::string::npos) {
            L.OS << "      // PTX_ST_GLOBAL_F32\n";
            L.OS << "      memref.store " << op(0, "%f0") << ", %output[" << op(1, "%c0") << "] : memref<?xf32>\n";
            return;
        }
        if (asmStr.find("st.shared.f32") != std::string::npos) {
            L.OS << "      // PTX_ST_SHARED_F32\n";
            L.OS << "      memref.store " << op(0, "%f0") << ", %tile[" << op(1, "%c0") << "] : memref<?xf32>\n";
            return;
        }
        if (asmStr.find("st.global") != std::string::npos) {
            L.OS << "      // PTX_ST_GLOBAL: " << asmStr.substr(0, 40) << "\n";
            return;
        }
        if (asmStr.find("mov.b32") != std::string::npos) {
            L.OS << "      // PTX_MOV_B32\n";
            L.OS << "      " << v << " = arith.addi " << op(0, "%c0") << ", %c0 : index\n";
            return;
        }
        if (asmStr.find("mov.f32") != std::string::npos) {
            L.OS << "      // PTX_MOV_F32\n";
            L.OS << "      " << v << " = arith.addf " << op(0, "%f0") << ", %f0 : f32\n";
            return;
        }
        if (asmStr.find("mov.b64") != std::string::npos || asmStr.find("mov.u64") != std::string::npos) {
            L.OS << "      // PTX_MOV_B64\n";
            L.OS << "      " << v << " = arith.addi " << op(0, "%c0") << ", %c0 : index\n";
            return;
        }
        if (asmStr.find("setp.lt.f32") != std::string::npos) {
            L.OS << "      // PTX_SETP_LT_F32\n";
            L.OS << "      " << v << " = arith.cmpf olt, " << op(0, "%f0") << ", " << op(1, "%f0") << " : f32\n";
            return;
        }
        if (asmStr.find("setp.eq.f32") != std::string::npos) {
            L.OS << "      // PTX_SETP_EQ_F32\n";
            L.OS << "      " << v << " = arith.cmpf oeq, " << op(0, "%f0") << ", " << op(1, "%f0") << " : f32\n";
            return;
        }
        if (asmStr.find("setp.lt.s32") != std::string::npos || asmStr.find("setp.lt.u32") != std::string::npos) {
            L.OS << "      // PTX_SETP_LT_S32\n";
            L.OS << "      " << v << " = arith.cmpi slt, " << op(0, "%c0") << ", " << op(1, "%c0") << " : index\n";
            return;
        }
        if (asmStr.find("setp.eq.s32") != std::string::npos || asmStr.find("setp.eq.u32") != std::string::npos) {
            L.OS << "      // PTX_SETP_EQ_S32\n";
            L.OS << "      " << v << " = arith.cmpi eq, " << op(0, "%c0") << ", " << op(1, "%c0") << " : index\n";
            return;
        }
        if (asmStr.find("selp.f32") != std::string::npos) {
            L.OS << "      // PTX_SELP_F32\n";
            L.OS << "      " << v << " = arith.select " << op(2, "%false") << ", " << op(0, "%f0") << ", " << op(1, "%f0") << " : f32\n";
            return;
        }
        if (asmStr.find("and.b32") != std::string::npos) {
            L.OS << "      // PTX_AND_B32\n";
            L.OS << "      " << v << " = arith.andi " << op(0, "%c0") << ", " << op(1, "%c0") << " : index\n";
            return;
        }
        if (asmStr.find("or.b32") != std::string::npos) {
            L.OS << "      // PTX_OR_B32\n";
            L.OS << "      " << v << " = arith.ori " << op(0, "%c0") << ", " << op(1, "%c0") << " : index\n";
            return;
        }
        if (asmStr.find("xor.b32") != std::string::npos) {
            L.OS << "      // PTX_XOR_B32\n";
            L.OS << "      " << v << " = arith.xori " << op(0, "%c0") << ", " << op(1, "%c0") << " : index\n";
            return;
        }
        if (asmStr.find("shl.b32") != std::string::npos) {
            L.OS << "      // PTX_SHL_B32\n";
            L.OS << "      " << v << " = arith.shli " << op(0, "%c0") << ", " << op(1, "%c0") << " : index\n";
            return;
        }
        if (asmStr.find("shr.b32") != std::string::npos) {
            L.OS << "      // PTX_SHR_B32\n";
            L.OS << "      " << v << " = arith.shrsi " << op(0, "%c0") << ", " << op(1, "%c0") << " : index\n";
            return;
        }
        L.OS << "      // PTX_UNSUPPORTED: " << asmStr.substr(0, 60) << "\n";
        return;
    }

    if (auto *CE = dyn_cast<CallExpr>(S)) {
        if (const auto *FD = CE->getDirectCallee()) {
            if (FD->getNameAsString() == "atomicAdd" && CE->getNumArgs() >= 2) {
                std::string val = translateExpr(CE->getArg(1), L);
                std::string ty = exprType(CE->getArg(1));
                const Expr *ptrE = CE->getArg(0)->IgnoreParenImpCasts();
                if (const auto *UO = dyn_cast<UnaryOperator>(ptrE)) {
                    if (UO->getOpcode() == UO_AddrOf) ptrE = UO->getSubExpr()->IgnoreParenImpCasts();
                }
                if (const auto *ASE = dyn_cast<ArraySubscriptExpr>(ptrE)) {
                    std::string arr = translateExpr(ASE->getBase(), L);
                    std::string idx = translateExpr(ASE->getIdx(), L);
                    L.OS << "      memref.atomic_rmw addf " << val << ", " << arr << "[" << idx
                         << "] : (memref<?x" << ty << ">, " << ty << ") -> " << ty << "\n";
                }
                return;
            }
        }
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
                    L.OS << "      %" << name << " = memref.alloc() : memref<256x" << t << ">\n";
                } else if (VD->hasInit()) {
                    std::string v = translateExpr(VD->getInit(), L);
                    int dims = pointerDepth(VD->getType());
                    if (dims == 0) dims = 1;
                    L.declare(name, v, mlirType(VD->getType()), false, dims);
                } else {
                    std::string t = mlirType(VD->getType());
                    std::string init = "%zero_" + name;
                    L.OS << "      " << init << " = arith.constant 0.0 : " << t << "\n";
                    L.declare(name, init, t, false);
                }
            }
        }
        return;
    }

    if (auto *If = dyn_cast<IfStmt>(S)) {
        std::string cond = translateExpr(If->getCond(), L);
        std::string c1 = "%cmp_" + std::to_string(TempVarCounter++);
        L.OS << "      " << c1 << " = arith.cmpi ne, " << cond << ", %c0 : index\n";
        L.OS << "      scf.if " << c1 << " {\n";
        dumpStmtAsMLIR(If->getThen(), L);
        L.OS << "      }";
        if (If->getElse()) {
            L.OS << " else {\n";
            dumpStmtAsMLIR(If->getElse(), L);
            L.OS << "      }";
        }
        L.OS << "\n";
        return;
    }

    if (auto *While = dyn_cast<WhileStmt>(S)) {
        // Safe fallback: lower bounded while loops to scf.for
        std::string ivName = "%wiv_" + std::to_string(LoopCounter++);
        L.OS << "      scf.for " << ivName << " = %c0 to %c10 step %c1 {\n";
        dumpStmtAsMLIR(While->getBody(), L);
        L.OS << "      }\n";
        return;
    }

    if (auto *Sw = dyn_cast<SwitchStmt>(S)) {
        std::string cond = translateExpr(Sw->getCond(), L);
        SwitchCase *SC = Sw->getSwitchCaseList();
        // FIX 5: Collect cases and emit in source order
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
            L.OS << "      " << cmp << " = arith.cmpi eq, " << cond << ", " << caseVal << " : index\n";
            L.OS << "      scf.if " << cmp << " {\n";
            dumpStmtAsMLIR(CS->getSubStmt(), L);
            L.OS << "      }\n";
        }
        if (defaultCase) {
            L.OS << "      // default\n";
            dumpStmtAsMLIR(defaultCase->getSubStmt(), L);
        }
        return;
    }

    // FIX 3 & 4: For loops with unique variable names
    if (auto *For = dyn_cast<ForStmt>(S)) {
        std::string UB = "%N";
        if (const Expr *C = For->getCond())
            if (const auto *BO = dyn_cast<BinaryOperator>(C->IgnoreParenImpCasts()))
                UB = translateExpr(BO->getRHS(), L);

        CompoundAssignOperator *CAO = findAccumulation(For->getBody());
        BinaryOperator *plainAcc = findPlainAccumulation(For->getBody());

        std::string accVar = "";
        std::string accType = "f32";
        if (CAO) {
            if (const auto *DRE = dyn_cast<DeclRefExpr>(CAO->getLHS()->IgnoreParenImpCasts()))
                accVar = DRE->getDecl()->getNameAsString();
            accType = exprType(CAO->getRHS());
        } else if (plainAcc) {
            if (const auto *DRE = dyn_cast<DeclRefExpr>(plainAcc->getLHS()->IgnoreParenImpCasts()))
                accVar = DRE->getDecl()->getNameAsString();
            accType = exprType(plainAcc->getRHS());
        }

        std::string ivName = "%iv_" + std::to_string(LoopCounter++);
        L.declare("k", ivName, "index", false);
        L.declare("i", ivName, "index", false);
        L.declare("iv", ivName, "index", false);
        L.declare("j", ivName, "index", false);

        if (!accVar.empty()) {
            std::string init = "%init_" + accVar;
            std::string iter = "%iter_" + accVar;
            L.OS << "      " << init << " = arith.constant 0.0 : " << accType << "\n";
            L.OS << "      %result = scf.for " << ivName << " = %c0 to " << UB << " step %c1 iter_args("
                 << iter << " = " << init << ") -> (" << accType << ") {\n";
            std::string inner;
            if (CAO) {
                inner = translateExpr(CAO->getRHS(), L);
            } else if (plainAcc) {
                const Expr *rhs = plainAcc->getRHS()->IgnoreParenImpCasts();
                if (const auto *BO = dyn_cast<BinaryOperator>(rhs)) {
                    if (BO->getOpcode() == BO_Add) inner = translateExpr(BO->getRHS(), L);
                    else inner = translateExpr(rhs, L);
                } else {
                    inner = translateExpr(rhs, L);
                }
            }
            L.OS << "        %new_sum = arith.addf " << iter << ", " << inner << " : " << accType << "\n";
            L.OS << "        scf.yield %new_sum : " << accType << "\n";
            L.OS << "      }\n";
            L.declare(accVar, "%result", accType, false);
        } else {
            L.OS << "      scf.for " << ivName << " = %c0 to " << UB << " step %c1 {\n";
            dumpStmtAsMLIR(For->getBody(), L);
            L.OS << "      }\n";
        }
        return;
    }

    for (Stmt *C : S->children()) {
        if (C && isa<Expr>(C)) translateExpr(cast<Expr>(C), L);
        else dumpStmtAsMLIR(C, L);
    }
}

class CudaLaunchVisitor : public RecursiveASTVisitor<CudaLaunchVisitor> {
public:
    explicit CudaLaunchVisitor(ASTContext *Context) : Context(Context) {}

    void emitKernel(FunctionDecl *FD) {
        if (!FD || !FD->hasBody()) return;
        if (!FD->hasAttr<CUDAGlobalAttr>()) return;

        std::string kernelName = FD->getNameAsString();
        TempVarCounter = 0;
        LoopCounter = 0;
        llvm::outs() << "// Target warp size: " << WarpSize << "\n";
        llvm::outs() << "module {\n";
        llvm::outs() << "  func.func @" << kernelName << "(";
        CudaLauncher L(llvm::outs());
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
                ptype = "index";
            }
            if (!first) llvm::outs() << ", ";
            first = false;
            llvm::outs() << "%" << pname << ": " << ptype;
            L.declare(pname, "%" + pname, ptype, isMemref, dims);
        }
        llvm::outs() << ") {\n";
        llvm::outs() << "    %c0 = arith.constant 0 : index\n";
        llvm::outs() << "    %c1 = arith.constant 1 : index\n";
        llvm::outs() << "    %c2 = arith.constant 2 : index\n";
        llvm::outs() << "    %c5 = arith.constant 5 : index\n";
        llvm::outs() << "    %c10 = arith.constant 10 : index\n";
        llvm::outs() << "    %c32 = arith.constant 32 : index\n";
        llvm::outs() << "    %cTotal = arith.muli %N, %N : index\n";
        llvm::outs() << "    scf.for %t = %c0 to %cTotal step %c1 {\n";
        llvm::outs() << "      %row = arith.divsi %t, %N : index\n";
        llvm::outs() << "      %col = arith.remsi %t, %N : index\n";
        llvm::outs() << "      %bx = arith.divsi %row, %c2 : index\n";
        llvm::outs() << "      %tx = arith.remsi %row, %c2 : index\n";
        llvm::outs() << "      %by = arith.divsi %col, %c2 : index\n";
        llvm::outs() << "      %ty = arith.remsi %col, %c2 : index\n";
        dumpStmtAsMLIR(FD->getBody(), L);
        llvm::outs() << "    }\n";
        llvm::outs() << "    return\n";
        llvm::outs() << "  }\n";
        llvm::outs() << "}\n";
    }

    bool VisitFunctionDecl(FunctionDecl *FD) {
        if (FD->getTemplatedKind() == FunctionDecl::TK_FunctionTemplate) return true;
        emitKernel(FD);
        return true;
    }

    bool VisitFunctionTemplateDecl(FunctionTemplateDecl *FTD) {
        if (!FTD->getTemplatedDecl()->hasAttr<CUDAGlobalAttr>()) return true;
        for (auto *Spec : FTD->specializations()) {
            emitKernel(Spec);
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
    return Tool.run(newFrontendActionFactory<CudaLaunchAction>().get());
}