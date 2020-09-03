---
layout: post
title: 位运算
date: 2020-09-03 
tags: leetcode   
---

### 常用位操作

在下面示例中，`1s` 和 `0s` 分别表示一串1和一串0

```
x ^ 0s = x      x & 0s = 0      x | 0s = x
x ^ 1s = ~x     x & 1s = x      x | 1s = 1s
x ^ x = 0       x & x = x       x | x = x
```


### 获取和设置数位

**1. 获取整型数值num二进制中的第i位。**

将1左移i位，得到形如 `00010000` 的值，对这个值与num执行“位与”操作，从而将i位之外的所有位清0，检查结果是否位0.

```c++
bool getBit(int num, int i) {
    return (num & (1 << i)) != 0;
}
```

**2. 将num的第i位设置位1**

```c++
int setBit(int num, int i) {
    return num | (1 << i);
}
```

**3. 将num的第i位清零**

将数字 `00 010 000` 取反，得到类似 `11 101 111` 数字，对该数字与num执行与操作。

```c++
int clearBit(int num, int i) {
    int mask = ~(1 << i);
    return num & mask;
}
```

如果要清零最高位至第i位（包括最高位和第i位），先创建一个第i位为1 (1 << i) 的源码，将其减1的到得到第一部分全为0，第二部分全为1的数字，再与目标数执行与操作

```c++
int claerBitsLeft(int num, int i) {
    int mask = (1 << i) - 1;
    return num & mask;
}
```

如果要清零第i位至第0位（包括第i位和第0位），使用一串1构成的数字(-1) ，将其做一 `i+1` 位，得到一串第一部分全为1，第二部分全为0的数字。

```c++
int clearBitsRight(int num, int i) {
    int mask = (-1 << (i + 1));
    return num & mask;
}
```

**4. 将num的第i位设置位v**

首先将num的第i位清零，然后将v左移i位，得到一个i位为1，其他位为0的数，最后将两个结果执行或操作。

```c++
int updateBit(int num, int i, bool bit) {
    int mask = ~(1 << i);
    int v = bit ? 1 : 0;
    return (num & mask) | (value << i);
}
```

### 一些有用的位操作

>* 判断整数n是否为2的某次方： `n & (n -1) == 0`  
>* 清除整数最右边的1：`n = n & (n - 1)`
>* 获得二进制中最低位的1: `lowbit = n & (-n)`
>* 得到一个数字的相反数(按位取反，再加一)： `n = (~n) + 1`


### 整数转换，确定需要改变几个位才能将整数A转成整数B。即对应二进制不同位置的数目（汉明距离）
示例：
>* input : 29 (11101), 15 (01111)
>* output : 2

```c++
int hammingDistance(int x, int y) {
    int n = x ^ y, cnt = 0;
    while (n) {
        n = n & (n - 1);
        cnt++;
    }
    return cnt;
}
```

### 不用额外变量交换整数的值


```c++
void swap(int &a, int &b) {
    a = a ^ b;
    b = a ^ b;
    a = a ^ b;
}
```

或者 
```c++
void swap(int &a, int &b) { // a = 3, b = 4
    a = a + b;   // a = 7, b = 4
    b = a - b;   // a = 7, b = 3
    a = a - b;   // a = 4, b = 3
}
```

### 只用位运算实现整数的加减乘除

```c++
int add(int a, int b) {
    int sum = a;
    while (b) {
        sum = a ^ b;
        b = (a & b) << 1;
        a = sum;
    }
    return sum;
}
```

```c++
int minus(int a, int b) {// a - b = a + (-b)
    return add(a, add(~b, 1));
}
```

**乘法**
a*b = a * 2^0 * b_0 + a * 2^1 * b_1 + ... + a * 2^31 * b_31

```c++
int multi(int a, int b) {
    int res = 0;
    while (b) {
        if ((b & 1) != 0)
            res = add(res, a);
        a <<= 1;
        b >>= 1;
    }
    return res;
}
```

**除法**

```c++
bool isNeg(int n) {
    return n < 0;
}

int getNeg(int n) {
    return add(~n, 1);
}

int div(int a, int b) {
    int x = isNeg(a) ? getNeg(a) : a;
    int y = isNeg(b) ? getNeg(b) : b;
    int res = 0;
    for (int i = 31; i > -1; i = minus(i, 1)) {
        if ((x >> 1) >= y) {
            res |= (1 << i);
            x = minus(x, y << i);
        }
    }
    return isNeg(a) ^ isNeg(b) ? getNeg(res) : res;
}
```