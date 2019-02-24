#pragma once


typedef unsigned char BOOL_TYPE;


#define DOUBLE_EQUALS       1E-4
#define MAX_PROGRAM_SIZE    2048
#define THREADS_IN_BLOCK    256
#define MAX_STACK_SIZE      1024



#define ARITY_VAR   0x00000000
#define VAR         0x00000000


#define ARITY_0     0x10000000
#define CONST       0x10000000


#define ARITY_1     0x20000000
#define NOT         0x20000000


#define ARITY_2     0x30000000
#define AND         0x30000000
#define OR          0x30000001
#define XOR         0x30000002
#define XNOR        0x30000003
#define NAND        0x30000004
#define NOR         0x30000005


#define ERR         0xFFFFFFFF
