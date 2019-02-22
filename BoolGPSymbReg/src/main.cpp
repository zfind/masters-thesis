#include <memory>
#include <ECF/ECF.h>
#include "Primitives.cpp"
#include "BenchmarkOp.h"

int main(int argc, char **argv) {
    StateP state(new State);

    // create tree genotype
    TreeP tree(new Tree::Tree);

    // create new functions and add them to function set
    Tree::PrimitiveP ifl(new If);
    tree->addFunction(ifl);
    Tree::PrimitiveP orp(new Or);
    tree->addFunction(orp);
    Tree::PrimitiveP andp(new And);
    tree->addFunction(andp);
    Tree::PrimitiveP notp(new Not);
    tree->addFunction(notp);
    Tree::PrimitiveP xorp(new Xor);
    tree->addFunction(xorp);
    Tree::PrimitiveP and2p(new And2);
    tree->addFunction(and2p);
    Tree::PrimitiveP xnorp(new XNor);
    tree->addFunction(xnorp);

    // custom type terminals
    for (uint i = 0; i < 20; i++) {
        Tree::PrimitiveP myTerm = (Tree::PrimitiveP) new BoolV;
        std::string name = "v" + uint2str(i);
        myTerm->setName(name);
        tree->addTerminal(myTerm);
    }

    // register genotype with our primitives
    state->addGenotype(tree);

    auto benchmarkOp = std::make_unique<BenchmarkOp>();
    state->setEvalOp(benchmarkOp.get());


    state->initialize(argc, argv);
    state->run();


    return 0;
}
