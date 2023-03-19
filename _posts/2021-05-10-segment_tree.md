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
- [cfedu题解](#cfedu题解)
  - [维护区间最值及出现次数](#维护区间最值及出现次数)
  - [维护区间最大子数组和](#维护区间最大子数组和)
  - [第k个1的下标](#第k个1的下标)
  - [根据逆序对恢复原始排列](#根据逆序对恢复原始排列)
  - [求嵌入区间的数目](#求嵌入区间的数目)
  - [求交叉区间的数目](#求交叉区间的数目)
  - [区间交替符号和](#区间交替符号和)
  - [区间逆序对](#区间逆序对)



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


## cfedu题解


### 维护区间最值及出现次数

[step1 c](https://codeforces.com/edu/course/2/lesson/4/1/practice/contest/273169/problem/C)

给定长度为n的数组a和m次操作，每次操作有如下两种形式:

+ 1 i v 赋值 a[i] = v.  (0 <= i < n, 0 <= v <= 1e9)
+ 2 l r 输出区间[l, r]中的最小值，以及区间中有多少个数等于最小值。

+ 1 <= n, m <= 1e5
+ 1 <= a[i] <= 1e9

**分析**

线段树维护两个值，一个区间最值，及出现次数。


```c++
#include <bits/stdc++.h>
using namespace std;
struct SegTree {
    ;// ...
};
struct S {
    int mn, cnt;
    S():mn(1e9), cnt(0) {}
    S(int m, int c): mn(m), cnt(c) {}
};

S op(S x, S y) {
    S res;
    int m1 = x.mn, c1 = x.cnt, m2 = y.mn, c2 = y.cnt;
    if (m1 < m2) {
        res = x;
    } else if (m1 > m2) {
        res = y;
    } else res = S(m1, c1 + c2);
    return res;
}
S e() {
    return S();
}
int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    
    int n, q;
    cin >> n >> q;
    vector<S> a(n);
    for (int i = 0; i < n; ++i) {
        cin >> a[i].mn;
        a[i].cnt = 1;   
    }
    SegTree<S, op, e> seg(a);
    for (int i = 0, op, x, y; i < q; ++i) {
        cin >> op >> x >> y;
        if (op == 1) {
            seg.set(x, S{y, 1});
        } else {
            S s = seg.get(x, y);
            cout << s.mn << ' ' << s.cnt << '\n';
        }
    }
    return 0;
}
```

### 维护区间最大子数组和

[setp2 a](https://codeforces.com/edu/course/2/lesson/4/2/practice/contest/273278/problem/A)

给定长度为n的数组a和m次(1 <= n, m <= 1e5)操作，每次操作给定i, v, 赋值a[i]=v. (0 <= i < n, -1e9 <= a[i], v <= 1e9).

输出m+1行，初始最大子数组和以及每次操作后的最大子数组和。


```c++
struct S {
    long long max_sum;
    long long all_sum;
    long long left_suf;
    long long right_pref;
    S(): max_sum(0), all_sum(0), left_suf(0), right_pref(0) {}
    S(long long ms, long long as, long long ls, long long rp) :
    max_sum(ms),all_sum(as),left_suf(ls),right_pref(rp) {}
};

S op(S x, S y) {
    S res;
    res.all_sum = x.all_sum + y.all_sum;
    res.left_suf = max(x.left_suf, x.all_sum + y.left_suf);
    res.right_pref = max(y.right_pref, y.all_sum + x.right_pref);
    res.max_sum = max({x.max_sum, y.max_sum, x.right_pref + y.left_suf});
    return res;
}

S e() {
    return S();
}
int main() {
    int n, q;
    cin >> n >> q;
    vector<S> a(n);
    long long x;
    for (int i = 0; i < n; ++i) {
        cin >> x;
        a[i] = S{x, x, x, x}; 
    }
    SegTree<S,op,e> seg(a);
    long long mx = max(0ll, seg.get_all().max_sum);
    cout << mx << "\n";
    for (int i = 0, op, x, y; i < q; ++i) {
        cin >> x >> y;
        seg.set(x, S{y, y,y,y});
        long long mx = max(0ll, seg.get_all().max_sum);
        cout << mx << "\n";
    }
}
```

### 第k个1的下标

[step2 B](https://codeforces.com/edu/course/2/lesson/4/2/practice/contest/273278/problem/B)


给定长度为n的数组a(a[i] = 0或1)和m次(1 <= n, m <= 1e5)操作，每次操作有如下两种形式:

+ 1 i 赋值 a[i] = 1 - a[i];
+ 2 k 输出第k个1的下标(保证k不超过1的数量，下标从0开始)

**分析**

线段树二分

```c++
using S = int;
S op(S x, S y) {
    S s = x + y;
    return s;
}
S e() {
    return S();
}
int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);

    int n, q;
    cin >> n >> q;
    vector<S> a(n);
    for (int i = 0; i < n; ++i) {
        cin >> a[i];
    }
    SegTree<S, op, e> seg(a);
    for (int i = 0, op, x, target; i < q; ++i) {
        cin >> op;
        if (op == 1) {
            cin >> x;
            seg.set(x, 1 - seg.get(x));
        } else {
            cin >> target;
            int p = seg.max_right(0, [&](int x) {
                return x < target + 1;
            });
            cout << p << "\n";
        }
    }
    return 0;
}
```

### 根据逆序对恢复原始排列

[step3 b](https://codeforces.com/edu/course/2/lesson/4/3/practice/contest/274545/problem/B)

对于长度为n(1<=n<=1e5)的排列p，给定数组a, a[i]表示 `j<i`且 `p[j]>p[i]`的数量，求最初排列。

**分析**

初始每个位置都是1，从右往左计算，每次从右往左二分，找到一个位置i，满足i右边有a[i]个1(表示有a[i]个比当前元素大的数还未使用). 然后将该位置置0，表示该数已使用。

```c++
using S = int;
S op(S x, S y) {
    S s = x + y;
    return s;
}
S e() {
    return S();
}
int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);

    int n;
    cin >> n;
    vector<int> a(n);
    for (int i = 0; i < n; ++i) {
        cin >> a[i];   
    }
    SegTree<S,op,e> seg(vector<int>(n, 1));
    vector<int> ans(n);
    for (int i = n - 1; i >= 0; i--) {
        int x =  seg.min_left(n, [&](int x) {
            return x < a[i] + 1;
        });
        seg.set(x - 1, 0);
        ans[i] = x;
    }
    for (int i = 0; i < n; ++i) {
        cout << ans[i] << " \n"[i == n];
    }
}
```

### 求嵌入区间的数目

[step3 c](https://codeforces.com/edu/course/2/lesson/4/3/practice/contest/274545/problem/C)

长度为`2*n, 1 <= n <= 1e5`的数组，1-n中每个数各出现2次，如果数字y的两次出现都在数字x的区间内，则称y是x的嵌入区间，对于1-n，分别求其嵌入区间的数目。

**分析**

遇到右边界时，将区间的左边界在线段树上的位置设为1，同时 统计坐标在左右边界之间出现的1的数目

```c++
using S = int;
int op(int x, int y) {
    return x + y;
}
int e(){
    return 0;
}
int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    int n;
    cin >> n;
    SegTree<S,op,e> seg(n*2);
    vector<int> a(n * 2), pos(n, -1), ans(n);
    for (int i = 0; i < n * 2; ++i) {
        cin >> a[i];
        a[i]--;
        if (pos[a[i]] == -1) pos[a[i]] = i;
        else {
            ans[a[i]] = seg.get(pos[a[i]], i + 1);
            seg.set(pos[a[i]], 1);
        }
    }
    for (int i = 0; i < n; ++i) {
        cout << ans[i] << " \n"[i == n - 1];
    }
}
```


### 求交叉区间的数目

[step3 d](https://codeforces.com/edu/course/2/lesson/4/3/practice/contest/274545/problem/D)


长度为`2*n, 1 <= n <= 1e5`的数组，1-n中每个数各出现2次，如果数字y的出现和数字x的出现位置存在交叉，则称y是x的交叉区间，对于1-n，分别求其嵌入区间的数目。

**分析**

交叉区间, ` * x * . x . ` 包括关于左边界的交叉和右边界的交叉，两者可以分别计算，以右边界交叉为例，

第一次遇到数字x时，将其位置置1，第二次遇到x时，统计两个x之间1的数目，同时将左边界置0.

对于左边界交叉，从后往前，相同方法再计算一遍即可。

```c++
using S = int;

int op(int x, int y) {
    return x + y;
}

int e(){
    return 0;
}

int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    int n;
    cin >> n;
    SegTree<S,op,e> seg(n*2);
    vector<int> a(n * 2), pos(n, -1), ans(n);
    for (int i = 0; i < n * 2; ++i) {
        cin >> a[i];
        a[i]--;
        if (pos[a[i]] == -1) {
            pos[a[i]] = i;
            seg.set(pos[a[i]], 1);
        }
        else {
            ans[a[i]] = seg.get(pos[a[i]] + 1, i);
            seg.set(pos[a[i]], 0);
        }
    }
    pos = vector<int>(n, -1);
    for (int i = 2 * n - 1; i >= 0; --i) {
        if (pos[a[i]] == -1) {
            pos[a[i]] = i;
            seg.set(pos[a[i]], 1);
        }
        else {
            ans[a[i]] += seg.get(i, pos[a[i]]);
            seg.set(pos[a[i]], 0);
        }
    }
    for (int i = 0; i < n; ++i) {
        cout << ans[i] << " \n"[i == n - 1];
    }
}
```

### 区间交替符号和

[step4 a](https://codeforces.com/edu/course/2/lesson/4/4/practice/contest/274684/problem/A)

长度为n的数组a和q次操作每次操作有两种：
1. 0, i, j 赋值 a[i] = j
2. 1, l, r 计算 `a[l] - a[l+1]+a[l+2]-...+/-a[r]`

+ 1 <= n, q <= 1e5
+ 1 <= i <= n, 1 <= j <= 1e4
+ 1 <= l <= r <= n

**分析**

可以用两个线段树，分别维护奇偶下标的和。

```c++
using S = int;
S op(S x, S y) {
    S s = x + y;
    return s;
}
S e() {
    return S();
}
int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    int n ,q;
    cin >> n;
    vector<int> a(n), b(n);
    for (int i = 0; i < n; ++i) {
        if (i % 2 == 0) cin >> a[i];
        else cin >> b[i];
    }

    SegTree<S, op, e> s1(a);
    SegTree<S, op, e> s2(b);
    cin >> q;
    while (q--) {
        int op,x,y;
        cin>>op>>x>>y;
        if(op==0){
            x--;
            if(x%2==0) s1.set(x,y);
            else s2.set(x,y);
        }else{
            x--;
            auto u = s1.get(x, y), v = s2.get(x, y);
            cout << (x % 2 == 0 ? u - v : v - u) << '\n';
        }
    }
    return 0;
}
```

### 区间逆序对

[step4 c](https://codeforces.com/edu/course/2/lesson/4/4/practice/contest/274684/problem/C)

长度为n的数组a `1<=a[i]<=40`, q个查询，每个查询有两种：
1. 1 x, y 求区间[x,y]中的逆序对数目, (1<=x<=y<=n)
2. 2 x y 赋值 a[x] = y (1<=x<=n, 1<=y<=40)

**分析**

数据范围较小，直接维护每个数的频率，合并时统计逆序对。

```c++
struct S{
    vector<int> freq;
    long long cnt;
    S():cnt(0),freq(40){}
};
S op(S x, S y) {
    S s;
    for (int i = 0; i < 40; ++i) {
        s.freq[i] = x.freq[i] + y.freq[i];
    }
    long long cnt = x.cnt + y.cnt;
    for (int i = 0; i < 40; ++i) {
        for (int j = i + 1; j < 40; ++j) {
            cnt += y.freq[i] * x.freq[j];
        }
    }
    s.cnt = cnt;
    return s;
}
S e() {
    return S();
}
int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    
    int n, q, x;
    cin >> n >> q;
    vector<S> a(n);
    for (int i = 0; i < n; ++i) {
        cin >> x;
        x--;
        a[i].freq[x] = 1;   
    }
    SegTree<S, op, e> seg(a);
    while (q--) {
        int op;
        cin >> op;
        if (op == 1) {
            int l, r;
            cin >> l >> r; l--;
            cout << seg.get(l, r).cnt << '\n';
        } else {
            int i, v;
            cin >> i >> v; i--;v--;
            S s; s.freq[v] = 1;
            seg.set(i, s);
        }
    }
    return 0;
}
```