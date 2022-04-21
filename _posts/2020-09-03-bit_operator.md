---
layout: post
title: 位运算
date: 2020-09-03 
tags: 面试算法   
---


===

Index
---
<!-- TOC -->

- [常用位操作](#常用位操作)
- [获取和设置数位](#获取和设置数位)
- [一些有用的位操作](#一些有用的位操作)
- [二进制所有是1的位](#二进制所有是1的位)
- [进制转换](#进制转换)
- [格雷编码](#格雷编码)
- [汉明距离](#汉明距离)
- [汉明权重](#汉明权重)
- [不用额外变量交换整数的值](#不用额外变量交换整数的值)
- [只用位运算实现整数的加减乘除](#只用位运算实现整数的加减乘除)

<!-- /TOC -->


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
>* << 左移乘二，>> 除以2， 1<<i = 2^i,  x >> i = x / 2^i.


### 二进制所有是1的位

例如: 13, 其二进制为 1101, 二进制中所有为1的位为0, 2, 3.
20， 其二进制为 10100, 二进制中所有为1的位为2, 4.

技巧： **对于任意在[0,35]中的k，2^k%37互不相等，且恰好取遍整数1-36**
利用这个性质可以使用hash代替取log运算，提高效率。

```c++
vector<int> getAll1(int n) {
    int H[37];
    for (int i = 0; i < 36; ++i) {
        H[(1ll << i) % 36] = i;
    }
    vector<int> res;
    while (n > 0) {
        res.push_back(H[(n & -n) % 37]);
        n -= lowbit(n);
    }
}
```
    
### 进制转换

1.将任意2-36进制数转化为10进制数

```c++
// s是给定的radix进制字符串
int Atoi(string s, int radix) {
    int ans = 0;
    for (int i = 0; i < s.size(); ++i) {
        auto t = s[i];
        if (t >= '0' && t <= '9') 
            ans = ans * radix + t - '0' + 10;
        else 
            ans = ans * radix + t - 'a' + 10;
    }
    return ans;
}
```

**strtol库函数**
函数原型为 `long int strtol(const char *nptr, char **endptr, int base)`
base是要转化的数的进制，非法字符会赋值给endptr，nptr是要转化的字符

例如：
```c++
string s = "10100";
char * stop;
int ans = strtol(s.c_str(), &stop, 2);
// int ans = strtol(s.c_str(), NULL, 2);
```

2.将10进制数转换为任意的radix进制数，结果为char型

```c++
//n是待转数字，radix是指定的进制
string intToA(int n, int radix) {
    string ans = "";
    do {
        int t = n % radix;
        if (t >= 0 && t <= 9) ans += t + '0';
        else ans += t - 10 + 'a';
        n /= radix;
    } while(n);
    reverse(ans.begin(), ans.end());
    return ans;
}
```

**itoi库函数**

可以将一个10进制数转换为任意的2-36进制字符串
函数原型：`char* itoa(int value, char* string, int radix)`
```c++
int num = 10;
char str[10];
itoa(num, str, 2); //将num转换为2进制，结果写在str中。
```

### 格雷编码

[leetcode 89](https://leetcode-cn.com/problems/gray-code/)

格雷编码是一个二进制数字系统，在该系统中，两个连续的数值仅有一个位数的差异。

给定一个代表编码总位数的非负整数 n，打印其格雷编码序列。即使有多个不同答案，你也只需要返回其中一种。

格雷编码序列必须以 0 开头。

**1.直接计算**

```
    关键是搞清楚格雷编码的生成过程, G(i) = i ^ (i/2);
    如 n = 3: 
    G(0) = 000, 
    G(1) = 1 ^ 0 = 001 ^ 000 = 001
    G(2) = 2 ^ 1 = 010 ^ 001 = 011 
    G(3) = 3 ^ 1 = 011 ^ 001 = 010
    G(4) = 4 ^ 2 = 100 ^ 010 = 110
    G(5) = 5 ^ 2 = 101 ^ 010 = 111
    G(6) = 6 ^ 3 = 110 ^ 011 = 101
    G(7) = 7 ^ 3 = 111 ^ 011 = 100
```

```c++
    vector<int> grayCode(int n) {
        vector<int> res(1 << n);
        for (int i = 0; i < (1 << n); ++i) {
            res[i] = i ^ (i >> 1);
        }
        return res;
    }
```

**2. DFS**

```c++
    vector<int> res;
    int vis[1<<16+2], m;
    void dfs(int s, int n) {
        if (res.size() == m) return;
        res.push_back(s);
        vis[s] = 1;
        for (int i = 0; i < n; ++i) {
            int c = s ^ (1 << i);
            if (!vis[c]) dfs(c, n);
        }
    }
    vector<int> grayCode(int n) {
        if (!n) return {0};
        memset(vis, 0, sizeof(vis));
        m = 1 << n;
        dfs(0, n);
        return res;
    }
```

### 汉明距离

整数转换，确定需要改变几个位才能将整数A转成整数B。即对应二进制不同位置的数目.

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

### 汉明权重

汉明权重是一串符号中不同于（定义在其所使用的字符集上的）零符号（zero-symbol）的个数。对于一个二进制数，它的汉明权重就等于它  的个数（即 popcount）。

```c++
int popcount(int x) {
    int cnt = 0;
    while (x) {
        cnt += x & 1;
        x >>= 1;
    }
    return cnt;
}
```

求一个数的汉明权重还可以使用 `lowbit` 操作：我们将这个数不断地减去它的 lowbit4，直到这个数变为 0。

```c++
int popcount(int x) {
    int cnt = 0;
    while (x) {
        cnt ++;
        x -= x & -x;
    }
    return cnt;
}
```

或者 使用GCC用于位运算的内建函数
` __builtin_popcount(unsigned int x)`


**构造汉明权重递增的排列**

在 状压 DP 中，按照 popcount 递增的顺序枚举有时可以避免重复枚举状态。这是构造汉明权重递增的排列的一大作用。

枚举 0 ~ n 按汉明权重递增的排列

```c++
    for (int i = 0; (1<<i)-1 <= n; i++) {
        for (int x = (1<<i)-1, t; x <= n; t = x+(x&-x), x = x ? (t|((((t&-t)/(x&-x))>>1)-1)) : (n+1)) {
            // 写下需要完成的操作
        }
    }
```

例如, 按照二进制为1的数目递增构造 0-15。

```c++
int n = 1 << 4 - 1; // 15
for (int i = 0; (1<<i)-1 <= n; i++) {
    cout <<"i = " << i << ": ";
    for (int x = (1<<i)-1, t; x <= n; t = x+(x&-x), x = x ? (t|((((t&-t)/(x&-x))>>1)-1)) : (n+1)) {
        cout << x << " ";
    }
    cout << "\n";
}
```

输出为：
```c++
i = 0: 0 
i = 1: 1 2 4 8 
i = 2: 3 5 6 
i = 3: 7
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
        b = (unsigned int)(a & b) << 1;
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