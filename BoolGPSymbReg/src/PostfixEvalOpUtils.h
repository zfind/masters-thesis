#pragma once

#include <cstdint>
#include <ECF/ECF.h>

using gp_code_t = uint32_t;
using gp_val_t = unsigned char;
using gp_fitness_t = uint32_t;

const double DOUBLE_EQUALS = 1E-4;
const int MAX_PROGRAM_SIZE = 2048;
const int THREADS_IN_BLOCK = 256;
const int MAX_STACK_SIZE = 1024;

const gp_code_t ARITY_V =   0x00000000;
const gp_code_t VAR =       0x00000000;

const gp_code_t ARITY_0 =   0x10000000;
const gp_code_t CONST =     0x10000000;

const gp_code_t ARITY_1 =   0x20000000;
const gp_code_t NOT =       0x20000000;

const gp_code_t ARITY_2 =   0x30000000;
const gp_code_t AND =       0x30000000;
const gp_code_t OR =        0x30000001;
const gp_code_t XOR =       0x30000002;
const gp_code_t XNOR =      0x30000003;
const gp_code_t NAND =      0x30000004;
const gp_code_t NOR =       0x30000005;

const gp_code_t ERR =       0xFFFFFFFF;

class PostfixEvalOpUtils {
public:

    static void ConvertToPostfix(IndividualP individual, char* programBuffer, int& programSize);

};