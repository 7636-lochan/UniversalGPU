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

using namespace clang;
using namespace clang::tooling;

static llvm::cl::OptionCategory MyToolCategory("cuda-launch-parser");
static int TempVarCounter = 0;

std::string mlirType(QualType T) {
    if (T->isFloatingType()) {
        if (T->isSpecificBuiltinType(BuiltinType::Half)) return "f16";
        if (T->isSpecificBuiltinType(BuiltinType::Double)) return "f64";
        return "f32";
    }
    if (T->isVectorType()) {
        const auto *VT = T->getAs<VectorType>();
        std::string elem = mlirType(VT->getElementType());
        unsigned n = VT->getNumElements();
        return "vector<" + std::to_string(n) + "x" + elem + ">";
    }
    return "index";
}

bool isFloatLike(QualType T) { return T->isFloatingType(); }
std::string exprType(const Expr *E) { return mlirType(E->getType()); }

struct Symbol {
    std::string mlirName;
    std::string type;
    bool isMemref;
};

class CudaLauncher {
public:
    std::map<std::string, Symbol> symbols;
    raw_ostream &OS;
    CudaLauncher(raw_ostream &OS) : OS(OS) {}
    void declare(const std::string &name, const std::string &mlirName,
                 const std::string &type, bool isMemref) {
        symbols[name] = {mlirName, type, isMemref};
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

    if (const auto *DRE = dyn_cast<DeclRefExpr>(E))
        return L.lookup(DRE->getDecl()->getNameAsString());

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
        std::string arr = translateExpr(ASE->getBase(), L);
        std::string idx = translateExpr(ASE->getIdx(), L);
        std::string v = "%" + std::to_string(TempVarCounter++);
        std::string elemType = exprType(E);
        
        if (elemType.find("vector<") != std::string::npos) {
            L.OS << "      " << v << " = vector.load " << arr << "[" << idx
                 << "] : memref<?x" << elemType << ">, " << elemType << "\n";
        } else {
            L.OS << "      " << v << " = memref.load " << arr << "[" << idx
                 << "] : memref<?x" << elemType << ">\n";
        }
        return v;
    }

    if (const auto *BO = dyn_cast<BinaryOperator>(E)) {
        if (BO->getOpcode() == BO_Assign) {
            const Expr *LHS = BO->getLHS()->IgnoreParenImpCasts();
            if (const auto *ASE = dyn_cast<ArraySubscriptExpr>(LHS)) {
                std::string arr = translateExpr(ASE->getBase(), L);
                std::string idx = translateExpr(ASE->getIdx(), L);
                std::string val = translateExpr(BO->getRHS(), L);
                std::string elemType = exprType(BO->getRHS());
                L.OS << "      memref.store " << val << ", " << arr << "[" << idx
                     << "] : memref<?x" << elemType << ">\n";
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
            case BO_LT:  op = flt ? "arith.cmpf olt," : "arith.cmpi slt,"; break;
            default: return lhs;
        }
        std::string v = "%" + std::to_string(TempVarCounter++);
        L.OS << "      " << v << " = " << op << " " << lhs << ", " << rhs << " : " << ty << "\n";
        return v;
    }

    if (const auto *CE = dyn_cast<CallExpr>(E)) {
        if (const auto *FD = CE->getDirectCallee()) {
            std::string name = FD->getNameAsString();
            if (name == "__shfl_sync" && CE->getNumArgs() >= 3) {
                std::string mask = translateExpr(CE->getArg(0), L);
                std::string val = translateExpr(CE->getArg(1), L);
                std::string src = translateExpr(CE->getArg(2), L);
                std::string v = "%" + std::to_string(TempVarCounter++);
                L.OS << "      " << v << " = nvvm.shfl.sync.down " << mask << ", " << val << ", " << src << " : f32\n";
                return v;
            }
        }
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

    if (auto *DS = dyn_cast<DeclStmt>(S)) {
        for (auto *D : DS->decls()) {
            if (auto *VD = dyn_cast<VarDecl>(D)) {
                std::string name = VD->getNameAsString();
                if (VD->hasAttr<CUDASharedAttr>()) {
                    std::string t = mlirType(VD->getType());
                    L.declare(name, "%" + name, t, true);
                    L.OS << "      %" << name << " = memref.alloc() : memref<256x" << t << ">\n";
                } else if (VD->hasInit()) {
                    std::string v = translateExpr(VD->getInit(), L);
                    L.declare(name, v, mlirType(VD->getType()), false);
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
        L.OS << "      scf.if " << cond << " {\n";
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
        std::string cond = translateExpr(While->getCond(), L);
        L.OS << "      scf.while " << cond << " {\n";
        dumpStmtAsMLIR(While->getBody(), L);
        L.OS << "      }\n";
        return;
    }

    if (auto *Sw = dyn_cast<SwitchStmt>(S)) {
        std::string cond = translateExpr(Sw->getCond(), L);
        SwitchCase *SC = Sw->getSwitchCaseList();
        while (SC) {
            if (isa<CaseStmt>(SC)) {
                auto *CS = cast<CaseStmt>(SC);
                std::string caseVal = translateExpr(CS->getLHS(), L);
                std::string cmp = "%" + std::to_string(TempVarCounter++);
                L.OS << "      " << cmp << " = arith.cmpi eq, " << cond << ", " << caseVal << " : index\n";
                L.OS << "      scf.if " << cmp << " {\n";
                dumpStmtAsMLIR(CS->getSubStmt(), L);
                L.OS << "      }\n";
            } else if (isa<DefaultStmt>(SC)) {
                auto *DS = cast<DefaultStmt>(SC);
                L.OS << "      // default\n";
                dumpStmtAsMLIR(DS->getSubStmt(), L);
            }
            SC = SC->getNextSwitchCase();
        }
        return;
    }

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

        if (!accVar.empty()) {
            std::string init = "%init_" + accVar;
            std::string iter = "%iter_" + accVar;
            L.OS << "      " << init << " = arith.constant 0.0 : " << accType << "\n";
            L.OS << "      %result = scf.for %iv = %c0 to " << UB << " step %c1 iter_args("
                 << iter << " = " << init << ") -> (" << accType << ") {\n";
            L.declare("k", "%iv", "index", false);
            std::string inner;
            if (CAO) {
                inner = translateExpr(CAO->getRHS(), L);
            } else if (plainAcc) {
                const Expr *rhs = plainAcc->getRHS()->IgnoreParenImpCasts();
                if (const auto *BO = dyn_cast<BinaryOperator>(rhs)) {
                    if (BO->getOpcode() == BO_Add) {
                        inner = translateExpr(BO->getRHS(), L);
                    } else {
                        inner = translateExpr(rhs, L);
                    }
                } else {
                    inner = translateExpr(rhs, L);
                }
            }
            L.OS << "        %new_sum = arith.addf " << iter << ", " << inner << " : " << accType << "\n";
            L.OS << "        scf.yield %new_sum : " << accType << "\n";
            L.OS << "      }\n";
            L.declare(accVar, "%result", accType, false);
        } else {
            L.OS << "      scf.for %iv = %c0 to " << UB << " step %c1 {\n";
            L.declare("k", "%iv", "index", false);
            dumpStmtAsMLIR(For->getBody(), L);
            L.OS << "      }\n";
        }
        return;
    }

    for (Stmt *C : S->children()) {
        if (C && isa<Expr>(C))
            translateExpr(cast<Expr>(C), L);
        else
            dumpStmtAsMLIR(C, L);
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
        llvm::outs() << "module {\n";
        llvm::outs() << "  func.func @" << kernelName << "(";
        CudaLauncher L(llvm::outs());
        bool first = true;
        for (auto *Param : FD->parameters()) {
            std::string pname = Param->getNameAsString();
            std::string ptype;
            bool isMemref = false;
            if (Param->getType()->isPointerType()) {
                QualType pointee = Param->getType()->getPointeeType();
                std::string elem = mlirType(pointee);
                ptype = "memref<?x" + elem + ">";
                isMemref = true;
            } else {
                ptype = "index";
            }
            if (!first) llvm::outs() << ", ";
            first = false;
            llvm::outs() << "%" << pname << ": " << ptype;
            L.declare(pname, "%" + pname, ptype, isMemref);
        }
        llvm::outs() << ") {\n";
        llvm::outs() << "    %c0 = arith.constant 0 : index\n";
        llvm::outs() << "    %c1 = arith.constant 1 : index\n";
        llvm::outs() << "    %c2 = arith.constant 2 : index\n";
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
    ClangTool Tool(ExpectedParser->getCompilations(), ExpectedParser->getSourcePathList());
    Tool.appendArgumentsAdjuster(clang::tooling::getInsertArgumentAdjuster(
        {"-x", "cuda", "--cuda-gpu-arch=sm_70", "--cuda-path=/tmp", "-nocudainc", "-nocudalib"},
        clang::tooling::ArgumentInsertPosition::BEGIN));
    return Tool.run(newFrontendActionFactory<CudaLaunchAction>().get());
}