#pragma once

#include <cstdint>

using gp_code_t = uint32_t;
using gp_val_t = double;
using gp_fitness_t = double;

const double DOUBLE_EQUALS = 1E-4;
const int MAX_PROGRAM_SIZE = 500;
const int THREADS_IN_BLOCK = 128;
const int MAX_STACK_SIZE = 500;

const gp_code_t ARITY_V =   0x00000000;
const gp_code_t VAR =       0x00000000;

const gp_code_t ARITY_0 =   0x10000000;
const gp_code_t CONST =     0x10000000;

const gp_code_t ARITY_1 =   0x20000000;
const gp_code_t SIN =       0x20000000;
const gp_code_t COS =       0x20000001;
const gp_code_t SQR =       0x20000002;

const gp_code_t ARITY_2 =   0x30000000;
const gp_code_t ADD =       0x30000000;
const gp_code_t SUB =       0x30000001;
const gp_code_t MUL =       0x30000002;
const gp_code_t DIV =       0x30000003;

const gp_code_t ERR =       0xFFFFFFFF;
