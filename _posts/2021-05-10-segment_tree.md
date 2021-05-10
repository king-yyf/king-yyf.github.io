---
layout: post
title: 线段树
date: 2021-05-10
tags: 算法模版  
---


===

Index
---
<!-- TOC -->

- [简介](#简介)
- [区间修改与懒标记](#区间修改与懒标记)
- [模板代码](#模板代码)
- [使用方法](#使用方法)
- [线段树练习题1](#线段树练习题1)



<!-- /TOC -->

### 简介

线段树是一种二叉搜索树，与区间树相似，它将一个区间划分成一些单元区间，每个单元区间对应线段树中的一个叶结点。使用线段树可以快速的查找某一个节点在若干条线段中出现的次数，时间复杂度为 O(log(n))。而未优化的空间复杂度为2N ，因此有时需要离散化让空间压缩。

用于 **幺半群** 的数据结构， (S, .: S x S -> S, e : S)，满足
- 分配律：(a · b )· c = a · (b · c)
- 存在单位元素，a⋅e = e⋅a = a 

**主要用途**

在O(log N)时间复杂度内实现：
- 单点修改
- 区间修改
- 区间查询，如区间求和，求区间最大值、最小值等
- 区间修改

线段树维护的信息，需要满足可加性，且要以可以接受的速度合并信息和修改信息，如果使用标记，标记也要满足可加性（例如取模就不满足可加性，对 4取模然后对 3 取模，两个操作就不能合并在一起做（事实上某些情况下可以暴力单点取模）

### 区间修改与懒标记

区间修改是个很有趣的东西……你想啊，如果你要修改区间 ，难道把所有包含在区间[l,r]中的节点都遍历一次、修改一次？那估计这时间复杂度估计会上天。这怎么办呢？
我们这里要引用一个叫做 **懒惰标记** 的东西。

### 模板代码

```c++
#include <vector>
#include <cassert>
using namespace std;

template <class S, S (*op)(S, S), S (*e)()>
struct segtree {
public:
    segtree() : segtree(0) {}
    explicit segtree(int n) : segtree(std::vector<S>(n, e())) {}
    explicit segtree(const std::vector<S>& v) : _n(int(v.size())) {
        log = ceil_pow2(_n);
        size = 1 << log;
        d = std::vector<S>(2 * size, e());
        for (int i = 0; i < _n; i++) d[size + i] = v[i];
        for (int i = size - 1; i >= 1; i--) {
            update(i);
        }
    }

    // @param n `0 <= n`
    // @return minimum non-negative `x` s.t. `n <= 2**x`
    int ceil_pow2(int n) {
        int x = 0;
        while ((1U << x) < (unsigned int)(n)) x++;
        return x;
    }

    void set(int p, S x) {
        assert(0 <= p && p < _n);
        p += size;
        d[p] = x;
        for (int i = 1; i <= log; i++) update(p >> i);
    }

    S get(int p) const {
        assert(0 <= p && p < _n);
        return d[p + size];
    }

    S prod(int l, int r) const {
        assert(0 <= l && l <= r && r <= _n);
        S sml = e(), smr = e();
        l += size;
        r += size;

        while (l < r) {
            if (l & 1) sml = op(sml, d[l++]);
            if (r & 1) smr = op(d[--r], smr);
            l >>= 1;
            r >>= 1;
        }
        return op(sml, smr);
    }

    S all_prod() const { return d[1]; }

    template <bool (*f)(S)> int max_right(int l) const {
        return max_right(l, [](S x) { return f(x); });
    }
    template <class F> int max_right(int l, F f) const {
        assert(0 <= l && l <= _n);
        assert(f(e()));
        if (l == _n) return _n;
        l += size;
        S sm = e();
        do {
            while (l % 2 == 0) l >>= 1;
            if (!f(op(sm, d[l]))) {
                while (l < size) {
                    l = (2 * l);
                    if (f(op(sm, d[l]))) {
                        sm = op(sm, d[l]);
                        l++;
                    }
                }
                return l - size;
            }
            sm = op(sm, d[l]);
            l++;
        } while ((l & -l) != l);
        return _n;
    }

    template <bool (*f)(S)> int min_left(int r) const {
        return min_left(r, [](S x) { return f(x); });
    }
    template <class F> int min_left(int r, F f) const {
        assert(0 <= r && r <= _n);
        assert(f(e()));
        if (r == 0) return 0;
        r += size;
        S sm = e();
        do {
            r--;
            while (r > 1 && (r % 2)) r >>= 1;
            if (!f(op(d[r], sm))) {
                while (r < size) {
                    r = (2 * r + 1);
                    if (f(op(d[r], sm))) {
                        sm = op(d[r], sm);
                        r--;
                    }
                }
                return r + 1 - size;
            }
            sm = op(d[r], sm);
        } while ((r & -r) != r);
        return 0;
    }

private:
    int _n, size, log;
    std::vector<S> d;

    void update(int k) { d[k] = op(d[2 * k], d[2 * k + 1]); }
};

```

### 使用方法

1.构造函数

```c++
segtree<S, op, e> seg(int n)
segtree<S, op, e> seg(vector<S> v)
```

需要定义
- S 类型
- S op(S a, S, b) 二元操作
- S e() 单位元素

例如，对于整型类型的区间最小值，定义方式如下

```c++
int op(int a, int b){
    return min(a, b);
}

int e() {
    //INF ,满足 op(e(), a) = op(a, e()) = a,  min(a, e()) = a
    return (int)1e9; 
}

segtree<int, op, e> seg(10); // 初始化长度为10数组，每个1元素都是 e()
```

- 约束： 0 <= n <= 1e8
- 时间复杂度 O(n)

2. set()函数

将 x 赋值给 a[p]

```c++
void seg.set(int p, int x)
```
- 约束: 0 <= p < n
- 时间复杂度 O(logn)

3.get函数

返回元素 a[p]

```c++
S seg.get(int p)
```
- 约束: 0 <= p < n
- 时间复杂度 O(1)

3.prod函数

如果 l=r, 返回 e(), 否则 返回 op(a[l], a[l+1], ... , a[r - 1]) 

```c++
S seg.prod(int l, int r)
```
- 约束: 0 <= l <= r <= n
- 时间复杂度 O(log(n))


4.all_prod函数

如果 n=0, 返回 e(), 否则 返回 op(a[0], a[1], ... , a[n - 1]) 

```c++
S seg.all_prod()
```

- 时间复杂度 O(1)


5.max_right函数

1. 需要定义 bool f(S x) 函数，max_right函数在线段树上执行二分查找
2. 应该定义以S为参数并返回bool的函数对象。

函数返回一个下标 r， 满足下面两个条件

- r = l or f(op(a[l], a[l+1], ..., a[r-1])) = true
- r = n or f(op(a[l], a[l+1], ..., a[r])) = false

如果 f 是单调的，
返回 满足 `f(op(a[l], a[l + 1], ..., a[r - 1])) = true` 的最大的 r

```c++
1. int seg.max_right<f>(int l)
2. int seg.max_right<F>(int l, F f)
```
-  0 <= l <= n
-  f(e()) = true
-  f函数传入相同的参数时，返回值相同
-  时间复杂度 O(logn)

6. min_left

```c++
(1) int seg.min_left<f>(int r)
(2) int seg.min_left<F>(int r, F f)
```

1. 需要定义 bool f(S x) 函数，max_right函数在线段树上执行二分查找
2. 应该定义以S为参数并返回bool的函数对象。

函数返回一个下标 l， 满足下面两个条件

- l = r or f(op(a[l], a[l + 1], ..., a[r - 1])) = true
- l = 0 or f(op(a[l - 1], a[l], ..., a[r - 1])) = false

如果 f 是单调的，
返回 满足 `f(op(a[l], a[l + 1], ..., a[r - 1])) = true` 的最小的 l

-  0 <= l <= n
-  f(e()) = true
-  f函数传入相同的参数时，返回值相同
-  时间复杂度 O(logn)

**不同信息的函数设置**

1.最大值
```c++
int op(int a, int b){
    return max(a, b);
}

int e() {
    return -1; // 比数组中最小元素要小 即满足 max(e(),a[i]) = a[i]
}
```

2.最小值
```c++
int op(int a, int b){
    return min(a, b);
}

int e() {
    //INF ,满足 op(e(), a) = op(a, e()) = a,  min(a, e()) = a
    return (int)1e9; 
}
```

3.区间和
```c++
int op(int a, int b){
    return a+b;
}

int e() {
    return 0;  // e() + a[i] = a[i]
}

//将第i个数+b，
seg.seg(a-1, seg.get(a-1) + b);
```



### 线段树练习题1

[atcoder pratice2](https://atcoder.jp/contests/practice2/tasks/practice2_j)

给一个长度为N的数组a，有Q个下面几种类型的查询

第i个查询的类型是Ti

- Ti = 1, 你将给两个数字 X, V, 将 a[X] 赋值为 V
- Ti = 2, 你将给两个数字 L, R, 计算 a[L], a[L+1], ..., a[R] 的最大值
- Ti = 3, 你将给两个数字 X, V, 计算最大的 j, 使得， X <= j <= N，V <= a[j],如果不存在者也的，返回 N+1;

**输入格式**

```
N Q
A1 A2 ... AN
query 1
...
query Q
```


**数据范围**

- 1 <= N <= 2e5
- 0 <= Ai <= 1e9
- 1 <= Q <= 1e5

```c++
#include <vector>
#include <iostream>
#include <cassert>
using namespace std;

template <class S, S (*op)(S, S), S (*e)()>
struct segtree {
    // ...
};

int op(int a, int b) {
    return max(a, b);
}

int e() { return -1;}

int target;
bool f(int v) {
    return v < target;
}

int main() {
   int n, q;
   cin >> n >> q;
   vector<int> a(n);
   for (int i = 0; i < n; ++i) cin >> a[i];
   
   segtree<int, op, e> seg(a);
   for (int i =0; i < q; ++i) {
       int t; cin >> t;
       if (t == 1) {
           int x, v; cin >> x >> v;
           seg.set(x - 1, v);
       } else if (t == 2) {
           int l, r; cin >> l >> r;
           cout << seg.prod(l - 1, r) << "\n"; 
       } else if (t == 3) {
           int p; cin >> p >> target;
           cout << seg.max_right<f>(p - 1) + 1 << "\n";
       }
   }
   return 0;
}
```
