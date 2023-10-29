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
- [单点修改例题](#单点修改例题)
  - [区间取模](#区间取模)
  - [区间加求区间GCD](#区间加求区间gcd)
  - [维护区间最值及出现次数](#维护区间最值及出现次数)
  - [维护区间最大子数组和](#维护区间最大子数组和)
  - [第k个1的下标](#第k个1的下标)
  - [根据逆序对恢复原始排列](#根据逆序对恢复原始排列)
  - [求嵌入区间的数目](#求嵌入区间的数目)
  - [求交叉区间的数目](#求交叉区间的数目)
  - [区间交替符号和](#区间交替符号和)
  - [区间逆序对](#区间逆序对)
  - [区间非下降子数组数目](#区间非下降子数组数目)
  - [区间方差](#区间方差)
- [带懒标记例题](#带懒标记例题)
  - [Lazy-区间取max](#区间取max)
  - [Lazy-区间赋值](#区间赋值)
  - [Lazy-区间加区间求min](#区间加区间求min)
  - [Lazy-区间乘与区间和](#区间乘与区间和)
  - [Lazy-区间OR与区间AND](#区间or与区间and)
  - [Lazy-区间赋值区间min](#区间赋值区间min)
  - [Lazy-区间赋值区间和](#区间赋值区间和)
  - [Lazy-区间异或区间和](#区间异或区间和)
  - [Lazy-区间加第一个大于x的数](#区间加第一个大于x的数)
  - [Lazy-区间赋值&区间加-求和](#区间赋值区间加-求和)
  - [Lazy-区间加等差数列](#区间加等差数列)
  - [Lazy-区间加求区间乘等差数列和](#区间加求区间乘等差数列和)
  - [Lazy-区间乘c加d](#区间乘c加d)
  - [Lazy-区间异或查询区间和](#区间异或查询区间和)
  - [Lazy-区间翻转求最长连续1数目](#区间翻转求最长连续1数目)
  - [Lazy-区间赋值取反查询最大连续1数目](#区间赋值取反查询最大连续1数目)
  - [Lazy-区间不同元素数目平方和](#区间不同元素数目平方和)
- [权值线段树](#权值线段树)
  - [统计大小在某个范围内的数量](#统计大小在某个范围内的数量)
  - [查询集合MEX](#查询集合mex)


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



## 单点修改例题

### 区间取模

[cf 438D](https://codeforces.com/contest/438/problem/D)

给定一个序列(长度n)和m次操作，每次操作有三种形式:
1. 计算 a[l,r]的和
2. 对 l<=i<=r, 执行 a[i] = a[i] % x
3. 赋值 a[k] = x

+ 1 <= n, m <= 1e5
+ 1 <= a[i] <= 1e9

**分析**

取模的性质 `a[i] % x <= a[i] / 2`， 即每执行一个取模操作，数值至少会减为原来的1半，n个数最多会执行`nlog(n)`次操作。维护区间最大值，最大值坐标和区间和，暴力模拟即可。

```c++
struct S {
    ll sum;
    int mx, pos;
    S(): sum(0),mx(-1),pos(-1){}
    S(ll a, int b, int c) :sum(a), mx(b), pos(c){}
};
 
S op(S x, S y) {
    S s;
    s.sum = x.sum + y.sum;
    s.mx = max(x.mx, y.mx);
    s.pos = x.mx > y.mx ? x.pos : y.pos;
    return s;
}
S e() {
    return S();
}
int main() {    
    int n, q, x;
    cin >> n >> q;
    vector<S> a(n);
    for (int i = 0; i < n; ++i) {
        cin >> x;
        a[i] = S{x, x, i};
    }
    SegTree<S, op, e> seg(a);
    for (int i = 0, t, l, r, p; i < q; ++i) {
        cin >> t;
        if (t == 1) {
            cin >> l >> r;
            l--;
            cout << seg.get(l, r).sum << '\n';
        } else if (t == 2){
            cin >> l >> r >> p;
            l--;
            auto s = seg.get(l, r);
            while (s.mx >= p) {
                seg.set(s.pos, S{s.mx % p, s.mx % p, s.pos});
                s = seg.get(l, r);
            }
        } else {
            cin >> p >> x;
            p--;
            seg.set(p, S{x, x, p});
        }
    }
    return 0;
}
```

### 区间加求区间gcd

[acwing 246](https://www.acwing.com/problem/content/description/247/)

给定长度为n的数组a和m次操作，每次操作有如下两种形式:
+ `C l r d` 把 a[l],..a[r] 都加上d
+ `Q l r` 输出 a[l],..,a[r]的最大公约数

+ 1 <= n <= 5e5
+ 1 <= m <= 1e5
+ 1 <= a[i], |d| <= 1e18

**分析**

`gcd(a[l],...a[r]) = gcd(a[l], a[l+1]-a[l], ..., a[r] - a[r - 1])`

设b为a的差分数组，则a数组区间加，可以转化为b数组两点加，`b[l]+d, b[r+1]-d`.
操作2:

```
gcd(a[l],a[l+1],..,a[r]) = gcd(a[l],a[l+1]-a[l], ..., a[r] - a[r - 1])
                         = gcd(a[l], gcd(b[l+1],b[l+2],...b[r]))
                         = gcd(b[1]+..+b[l], gcd(b[l+1],b[l+2],...b[r]))
```

所以，只需用线段树维护差分数组的区间和以及区间gcd即可。

```c++
struct S{
    long long s,d; //差分数目区间和，区间GCD
    S& operator + (const long long x) {
        s += x, d += x;
        return *this;
    }
};
S op(S x, S y) {
    return S{x.s + y.s,gcd(x.d, y.d)};
}
S e() {
    return S{0,0};
}
void ac_yyf(int tt) {
    cin >> n >> m;
    vector<long long> a(n + 1);
    for (int i = 1; i <= n; ++i) {
        cin >> a[i];
    }
    vector<S> v(n + 1);
    for (int i = 0; i < n; ++i) {
        v[i] = S{a[i + 1] - a[i], a[i + 1] - a[i]};
    }
    SegTree<S, op, e> seg(v);

    char c;
    long long d;
    for (int i = 0, l, r; i < m; ++i) {
        cin >> c >> l >> r;
        if (c == 'C') {
            cin >> d;
            l--;
            seg.set(l, seg.get(l) + d);
            seg.set(r, seg.get(r) + (-d));
        } else {
            cout << gcd(seg.get(0, l).s, seg.get(l, r).d) << '\n';
        }
    }
}
```

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

### 区间非下降子数组数目

[no inversioins](https://www.hackerearth.com/practice/data-structures/advanced-data-structures/segment-trees/practice-problems/algorithm/no-inversions-db0ebea5/)

给定长度为n的字符串s，仅包含小写字母，q次询问，每次询问给定[l,r],求区间[l,r]中优秀子串的数目，如果一个字符串不存在逆序对，则称优秀子串。

+ 1 <= n, q <= 2e5
+ l <= l <= r <= n

```c++
struct S {
    long long cnt; //总数目
    int lcnt, rcnt; // 从左边/右边非递减最大长度
    char left, right;  // 最左边和最右边字符
    bool isGood;
    S():cnt(0),lcnt(0),rcnt(0),left(0),right(0),isGood(true){}
    S(int u){
        cnt=lcnt=rcnt=(u>0);
        left=right=u; isGood=true;
    }
};
 
S op(S x, S y) {
    S s;
    s.cnt = x.cnt + y.cnt;
    if (x.right <= y.left) {
        s.cnt += x.rcnt * 1ll * y.lcnt;
        s.isGood = x.isGood & y.isGood;
        s.lcnt = x.isGood ? y.lcnt + x.lcnt : x.lcnt;
        s.rcnt = y.isGood ? x.rcnt + y.rcnt : y.rcnt;
    } else {
        s.isGood = false;
        s.lcnt = x.lcnt;
        s.rcnt = y.rcnt;
    }
    s.left = x.left;
    s.right = y.right;
    return s;
}
S e() {
    return S{};
}
void ac_yyf(int tt) {
    rd(n,s,q);
    SegTree<S, op, e> seg(n);
    for (int i = 0; i < n; ++i) {
        seg.set(i, s[i]-'a'+1);
    }
    while(q--){
        rd(x,y);
        cout<<seg.get(x-1,y).cnt<<nl;
    }
}
```

### 区间方差

[洛谷 p5142](https://www.luogu.com.cn/problem/P5142)

给定一个长为n的数组吗，m次查询。

+ 1 l r 赋值 a[l] = r
+ 2 l r 求区间 [l,r]的方差, 模1e9+7

+ 1 <= n,m <= 1e5
+ 1 <= a[i], y <= 1e9

```c++
using S = mint;
S op(S x, S y) {
    return x + y;
}
S e() {
    return S();
}
int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    
    int n, m;
    cin >> n >> m;
    vector<mint> a(n), b(n);
    for (int i = 0; i < n; ++i) {
        cin >> a[i];
        b[i] = a[i] * a[i];
    }

    SegTree<S, op, e> s1(a), s2(b);

    for (int i = 0, t, x, y; i < m; ++i) {
        cin >> t >> x >> y;
        x--;
        if (t == 1) {
            s1.set(x, y);
            s2.set(x, mint(y) * y);
        } else {
            mint c = s1.get(x, y) / (y - x);
            mint ans =s2.get(x, y) / (y - x) - c * c;
            cout << ans << '\n';
        }
    }
    return 0;
}

```

## 带懒标记例题


### 区间取max

[part2 step1B](https://codeforces.com/edu/course/2/lesson/5/1/practice/contest/279634/problem/B)


n个初始为0元素，m次操作。(1<=n,m<=1e5)
+ 1 l r v : 对所有 l <= i < r 执行 a[i] = max(a[i], v) (0 <= v, a[i] <= 1e9)
+ 2 i 输出第i个元素

```c++
struct S {
    int mx;
    int size;
};
using F = int;
S op(S x, S y) {
    S s;
    s = S{max(x.mx, y.mx), x.size + y.size};
    return s;
}
S e() {
    return S();
};
S tag(F f, S s) { 
    S res;
    res.mx = max(f, s.mx);
    res.size = s.size;
    return res;
}
F merge(F x, F y) { return max(x, y); }
F id() { return 0; }

int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    
    int n, m;
    cin >> n >> m;
    LazySegTree<S, op, e, F, tag, merge, id> seg(vector<S>(n,{0,1}));
    for (int i = 0, op, l, r, v; i < m; ++i) {
        cin >> op;
        if (op == 1) {
            cin >> l >> r >> v;
            seg.apply(l, r, v);
        } else {
            cin >> v;
            cout << seg.get(v).mx << '\n';
        }
    }
    return 0;
}
```

### 区间赋值

[part2 step1C](https://codeforces.com/edu/course/2/lesson/5/1/practice/contest/279634/problem/C)

n个初始为0元素，m次操作。(1<=n,m<=1e5)
+ 1 l r v : 对所有 l <= i < r 执行 a[i] = v (0 <= v <= 1e9)
+ 2 i 输出第i个元素

```c++
struct S {
    long long sum;
    int size;
};
using F = long long;
S op(S x, S y) {
    S s;
    s = S{x.sum + y.sum, x.size + y.size};
    return s;
}
S e() {
    return S();
};
S tag(F f, S s) { 
    S res;
    res.size = s.size;
    res.sum = f == -1 ? s.sum : f * s.size;
    return res;
}
F merge(F x, F y) { 
    return x == -1 ? y : x;
}
F id() { return -1; }  // -1表示无懒标记

int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    
    int n, m;
    cin >> n >> m;
    LazySegTree<S, op, e, F, tag, merge, id> seg(vector<S>(n,{0,1}));
    for (int i = 0, op, l, r, v; i < m; ++i) {
        cin >> op;
        if (op == 1) {
            cin >> l >> r >> v;
            seg.apply(l, r, v);
        } else {
            cin >> v;
            cout << seg.get(v).sum << '\n';
        }
    }

    return 0;
}
```

### 区间加区间求min

[part2 step2A](https://codeforces.com/edu/course/2/lesson/5/2/practice/contest/279653/problem/A)

n个初始为0元素，m次操作。(1<=n,m<=1e5)
+ 1 l r v : 对所有 l <= i < r 执行 a[i] += v (0 <= v <= 1e9)
+ 2 l r 输出min(a[l], a[l+1], ...a[r-1])

```c++
struct S {
    long long sum, mn;
    int size;
    S():sum(0),mn(1e18),size(0){} // mn 默认取最大值
    S(long long s, long long m, int siz):sum(s),mn(m),size(siz){}
};
using F = long long;
S op(S x, S y) {
    return S{x.sum + y.sum, min(x.mn, y.mn), x.size + y.size};
}
S e() {
    return S();
};
S tag(F f, S s) { 
    return S{s.sum + f * s.size, s.mn + f, s.size};
}
F merge(F x, F y) { 
    return x + y;
}
F id() { return 0; }  

int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    
    int n, m;
    cin >> n >> m;
    LazySegTree<S, op, e, F, tag, merge, id> seg(vector<S>(n,{0,0,1}));
    for (int i = 0, op, l, r, v; i < m; ++i) {
        cin >> op;
        if (op == 1) {
            cin >> l >> r >> v;
            seg.apply(l, r, v);
        } else {
            cin >> l >> r;
            cout << seg.get(l, r).mn << '\n';
        }
    }
    return 0;
}
```

### 区间乘与区间和

[part2 step2B](https://codeforces.com/edu/course/2/lesson/5/2/practice/contest/279653/problem/B)


n个初始为1元素，m次操作。(1<=n,m<=1e5)
+ 1 l r v : 对所有 l <= i < r 执行 a[i] *= v (1 <= v <= 1e9+7)
+ 2 l r 输出sum(a[l...r-1])的和模1e9+7

```c++
struct S {
    mint s;
    int size;
    S():s(0),size(0){}
    S(mint _s, int siz):s(_s),size(siz){}
};
using F = mint;
S op(S x, S y) {
    return S{x.s + y.s, x.size + y.size};
}
S e() {
    return S();
};
S tag(F f, S s) { 
    return S{s.s * f, s.size};
}
F merge(F x, F y) { 
    return x * y;
}
F id() { return 1; } 

int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    
    int n, m;
    cin >> n >> m;
    LazySegTree<S, op, e, F, tag, merge, id> seg(vector<S>(n,{1,1}));
    for (int i = 0, op, l, r, v; i < m; ++i) {
        cin >> op;
        if (op == 1) {
            cin >> l >> r >> v;
            seg.apply(l, r, v);
        } else {
            cin >> l >> r;
            cout << seg.get(l, r).s << '\n';
        }
    }
    return 0;
}
```

### 区间or与区间and

[part2 step2C](https://codeforces.com/edu/course/2/lesson/5/2/practice/contest/279653/problem/C)


n个初始为0元素，m次操作。(1<=n,m<=1e5)
+ 1 l r v : 对所有 l <= i < r 执行 `a[i] |= v (1 <= v <= 2^30)`
+ 2 l r 输出AND(a[l...r-1])的和

```c++
struct S {
    int sum;
    int size;
    S():sum(((1<<30)-1)),size(0){} // and e()默认值需要为全为1的数。
    S(int s, int siz):sum(s),size(siz){}
};
using F = int;
S op(S x, S y) {
    return S{x.sum & y.sum, x.size + y.size};
}
S e() {
    return S();
};
S tag(F f, S s) { 
    return S{s.sum | f, s.size};
}
F merge(F x, F y) { 
    return x | y;
}
F id() { return 0; } 

int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    
    int n, m;
    cin >> n >> m;
    LazySegTree<S, op, e, F, tag, merge, id> seg(vector<S>(n,{0,1}));
    for (int i = 0, op, l, r, v; i < m; ++i) {
        cin >> op;
        if (op == 1) {
            cin >> l >> r >> v;
            seg.apply(l, r, v);
        } else {
            cin >> l >> r;
            cout << seg.get(l, r).sum << '\n';
        }
    }
    return 0;
}
```

### 区间赋值区间min

[part2 step2E](https://codeforces.com/edu/course/2/lesson/5/2/practice/contest/279653/problem/E)


n个初始为0元素，m次操作。(1<=n,m<=1e5)
+ 1 l r v : 对所有 l <= i < r 执行 a[i] = v (1 <= v <= 1e9)
+ 2 l r 输出min(a[l...r-1])

```c++
struct S {
    long long sum, mn;
    S():sum(0),mn(1e18){} // mn初始化1e18,这里可以不用维护size
    S(long long s, long long m):sum(s),mn(m){}
};
using F = long long;
S op(S x, S y) {
    return S{x.sum + y.sum, min(x.mn, y.mn)};
}
S e() {
    return S();
};
S tag(F f, S s) { 
    return f == -1 ? S{s.sum, s.mn} : S(f, f);
}
F merge(F x, F y) { 
    return x == -1 ? y : x;
}
F id() { return -1; }  // -1: no tag
LazySegTree<S, op, e, F, tag, merge, id> seg(vector<S>(n,{0,0}));
```

### 区间赋值区间和

[part2 step2F](https://codeforces.com/edu/course/2/lesson/5/2/practice/contest/279653/problem/F)


n个初始为0元素，m次操作。(1<=n,m<=1e5)
+ 1 l r v : 对所有 l <= i < r 执行 a[i] = v (1 <= v <= 1e9)
+ 2 l r 输出sum(a[l...r-1])

```c++
struct S {
    long long sum;
    int size;
    S():sum(0),size(0){}
    S(long long s, int siz):sum(s),size(siz){}
};
using F = long long;
S op(S x, S y) {
    return S{x.sum + y.sum, x.size+y.size};
}
S e() {
    return S();
};
S tag(F f, S s) { 
    return f == -1 ? S{s.sum, s.size} : S{f * s.size, s.size};
}
F merge(F x, F y) { 
    return x == -1 ? y : x;
}
F id() { return -1; }  // -1: no tag
LazySegTree<S, op, e, F, tag, merge, id> seg(vector<S>(n,{0,1}));
```

### 区间赋值区间最大子数组

[part2 step3A](https://codeforces.com/edu/course/2/lesson/5/3/practice/contest/280799/problem/A)

n个初始为0元素，m次操作。(1<=n,m<=1e5),每次操作
+ l r v : 对所有 l <= i < r 执行 a[i] = v (-1e9 <= v <= 1e9)

每次操作输出最大子数组和。


```c++
struct S {
    long long max_sum, sum, max_pre, max_suf;
    int size;
    S(): max_sum(0), sum(0), max_pre(0), max_suf(0), size(0){}
    S(long long ms, long long s, long long pre, long long suf, int siz) :
    max_sum(ms),sum(s),max_pre(pre),max_suf(suf), size(siz){}
};

S op(S x, S y) {
    S res;
    res.sum = x.sum + y.sum;
    res.size = x.size + y.size;
    res.max_pre = max(x.max_pre, x.sum + y.max_pre);
    res.max_suf = max(y.max_suf, y.sum + x.max_suf);
    res.max_sum = max({x.max_sum, y.max_sum, x.max_suf + y.max_pre});
    return res;
}
const long long  inf = 1e18;
using F = long long;
S e() {
    return S();
};
S tag(F f, S s) { 
    if (f == inf) return s;
    long long p = f * s.size;
    return f > 0 ? S{p, p, p, p, s.size} : S{0, p, 0, 0, s.size};
}
F merge(F x, F y) { 
    return x == inf ? y : x;
}
F id() { return inf; }  // v的范围在[-1e9,1e9],需要找一个区间外的数作为无懒标记标志

int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    
    int n, m;
    cin >> n >> m;
    LazySegTree<S, op, e, F, tag, merge, id> seg(vector<S>(n,{0,0,0,0,1}));
    for (int i = 0, l, r, v; i < m; ++i) {
        cin >> l >> r >> v;
        seg.apply(l, r, v);
        cout << seg.get_all().max_sum << '\n';
    }
    return 0;
}
```

### 区间异或区间和

[part2 step3B](https://codeforces.com/edu/course/2/lesson/5/3/practice/contest/280799/problem/B)

n个初始为0元素，m次操作。(1<=n,m<=1e5),每次操作
+ 1 l r : 对所有 l <= i < r 执行 a[i] ^ 1
+ 2 k : 输出第k个1下标 (k从0开始，下标从0开始)


```c++
struct S {
    int sum, size;
    S():sum(0),size(0){}
    S(int s, int siz): sum(s), size(siz){}
};
using F = int;
S op(S x, S y) {
    return S{x.sum + y.sum, x.size + y.size};
}
S e() {
    return S();
};
S tag(F f, S s) { 
    return f == 0 ? s : S{s.size - s.sum, s.size};
}
F merge(F x, F y) { 
    return x ^ y;
}
F id() { return 0; }  // 0: no tag

int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    
    int n, m;
    cin >> n >> m;
    LazySegTree<S, op, e, F, tag, merge, id> seg(vector<S>(n,{0,1}));
    for (int i = 0, op, l, r, k; i < m; ++i) {
        cin >> op;
        if (op == 1) {
            cin >> l >> r;
            seg.apply(l, r, 1);
        } else {
            cin >> k;
            r = seg.max_right(0, [&](S x){
                return x.sum < k + 1;
            });
            cout << r << '\n';
        }
    }
    return 0;
}
```

### 区间加第一个大于x的数

[part2 step3C](https://codeforces.com/edu/course/2/lesson/5/3/practice/contest/280799/problem/C)

n个初始为0元素，m次操作。(1<=n,m<=1e5),每次操作
+ 1 l r x : 对所有 l <= i < r 执行 a[i] += x
+ 2 x l : 查询第一个位置 j >= l，满足 a[j] >= x

**分析**

维护区间和，区间最大值，操作2时在线段树上进行二分。

```c++
struct S {
    long long sum, mx;
    int size;
    S():sum(0),mx(0),size(0){} 
    S(long long s, long long m, int siz):sum(s),mx(m),size(siz){}
};
using F = long long;
S op(S x, S y) {
    return S{x.sum + y.sum, max(x.mx, y.mx), x.size + y.size};
}
S e() {
    return S();
};
S tag(F f, S s) { 
    return S{s.sum + f * s.size, s.mx + f, s.size};
}
F merge(F x, F y) { 
    return x + y;
}
F id() { return 0; }  

int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    
    int n, m;
    cin >> n >> m;
    LazySegTree<S, op, e, F, tag, merge, id> seg(vector<S>(n,{0,0,1}));
    for (int i = 0, op, l, r, x; i < m; ++i) {
        cin >> op;
        if (op == 1) {
            cin >> l >> r >> x;
            seg.apply(l, r, x);
        } else {
            cin >> x >> l;
            r = seg.max_right(l, [&](S s){
                return s.mx < x;
            });
            cout << (r == n ? -1 : r) << '\n';
        }
    }
    return 0;
}
```

### 区间赋值区间加-求和

[part2 step4a](https://codeforces.com/edu/course/2/lesson/5/4/practice/contest/280801/problem/A)

n个初始为0元素，m次操作。(1<=n,m<=1e5),每次操作
+ 1 l r x : 对所有 l <= i < r 执行 a[i] = x
+ 2 l r x : 对所有 l <= i < r 执行 a[i] = a[i] + x
+ 3 l r : 求 sum[l...r-1]


```c++
struct S {
    long long sum;
    int size;
    S():sum(0),size(0){} 
    S(long long s, int siz):sum(s),size(siz){}
};

struct F{
    int t;  // t=0: 赋值， t=1 加和
    long long v;
};

S op(S x, S y) {
    if (x.size == 0) return y; if (y.size == 0) return x;
    return S{x.sum + y.sum, x.size + y.size};
}
S e() {
    return S();
};
S tag(F f, S s) { 
    if (f.t == 1 && f.v == 0) return s;
    return S{(f.t ? s.sum : 0) + s.size * f.v, s.size};
}
F merge(F x, F y) { 
    return x.t == 0 ? x : F{y.t, y.v + x.v};
}
F id() { return F{1, 0}; }  // 

int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    
    int n, m;
    cin >> n >> m;
    LazySegTree<S, op, e, F, tag, merge, id> seg(vector<S>(n,{0,1}));
    for (int i = 0, op, l, r, x; i < m; ++i) {
        cin >> op;
        if (op == 1) {
            cin >> l >> r >> x;
            seg.apply(l, r, F{0, x});
        } else if (op == 2){
            cin >> l >> r >> x;
            seg.apply(l, r, F{1, x});
        } else {
            cin >> l >> r;
            cout << seg.get(l, r).sum << '\n';
        }
    }
    return 0;
}
```

### 区间加等差数列

[part2 step4b](https://codeforces.com/edu/course/2/lesson/5/4/practice/contest/280801/problem/B)


n个初始为0元素，m次操作。(1<=n,m<=2e5),每次操作
+ 1 l r x d : 对所有 l <= i < r 执行 a[i] = a[i] + x + (i - l) * d (1 <= l <= r <= n)
+ 2 x : 输出a[x] (1 <= x <= n)

+ 1 <= x, d <= 2e5

**分析**

在正常区间加的基础上，可以认为第i个位置的size为i，这样在求和时不同位置增加为等差数列，领一个线段树维护每个位置需要减去多少，在处理[l,r]区间时，以下标l为例，第一个线段树维护的时 `a[l]=a[l]+l*d` 第二个线段树维护 `a[l]+=a-l*d` 两个线段树相减即为要求的结果。

```c++
struct S {
    long long sum;
    int size;
    S():sum(0),size(0){} 
    S(long long s, int siz):sum(s),size(siz){}
};
using F = long long;
S op(S x, S y) {
    return S{x.sum + y.sum, x.size + y.size};
}
S e() {
    return S();
};
S tag(F f, S s) { 
    return S{s.sum + f * s.size, s.size};
}
F merge(F x, F y) { 
    return x + y;
}
F id() { return 0; }  

int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    
    int n, m;
    cin >> n >> m;
    vector<S> v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = {0, i};
    } 
    LazySegTree<S, op, e, F, tag, merge, id> seg(v);
    LazySegTree<S, op, e, F, tag, merge, id> seg2(vector<S>(n,{0,1}));
    for (int i = 0, op, l, r, a, d; i < m; ++i) {
        cin >> op;
        if (op == 1) {
            cin >> l >> r >> a >> d;
            l--;
            seg.apply(l, r, d);
            seg2.apply(l, r, a - d * 1ll * l);
        } else {
            cin >> l;
            l--;
            cout << seg.get(l).sum + seg2.get(l).sum << '\n';
        }
    }
    return 0;
}
```

### 区间加求区间乘等差数列和

[part2 step4d](https://codeforces.com/edu/course/2/lesson/5/4/practice/contest/280801/problem/D)


给定长度为n个数组a (-100 < a[i] < 100)，m次操作。(1<=n,m<=2e5),每次操作
+ 1 l r x: 对所有 l <= i < r 执行 a[i] = a[i] + x (1 <= l <= r <= n, -100 < x < 100)
+ 2 l r : 输出 a[l] * 1 + a[l + 1] * 2 + ... + a[r] * (r - l + 1)


```c++
struct S {
    long long sum;
    long long size;
    S():sum(0),size(0){} 
    S(long long s, long long siz):sum(s),size(siz){}
};
using F = long long;
S op(S x, S y) {
    return S{x.sum + y.sum, x.size + y.size};
}
S e() {
    return S();
};
S tag(F f, S s) { 
    return S{s.sum + f * s.size, s.size};
}
F merge(F x, F y) { 
    return x + y;
}
F id() { return 0; }  

int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    
    int n, m;
    cin >> n >> m;
    vector<S> a(n), b(n);
    for (int i = 0, x; i < n; ++i) {
        cin >> x;
        a[i] = {(i + 1) * 1ll * x, i + 1};
        b[i] = {x, 1};
    } 
    LazySegTree<S, op, e, F, tag, merge, id> seg(a), seg2(b);
    for (int i = 0, op, l, r, d; i < m; ++i) {
        cin >> op;
        if (op == 1) {
            cin >> l >> r >> d;
            l--;
            seg.apply(l, r, d);
            seg2.apply(l, r, d);
        } else {
            cin >> l >> r;
            l--;
            cout << seg.get(l, r).sum - l * seg2.get(l, r).sum << '\n';
        }
    }
    return 0;
}
```

### 区间乘c加d

[judge range_affine_point_get](https://judge.yosupo.jp/problem/range_affine_point_get)

给定长度为n的数组a, q次操作，
1. 0 l c c d  对所有 l <= i < r 执行 a[i] = a[i] * c + d;
2. 1 x 输出 a[x] % 998244353

+ 1 <= n, q <= 5e5
+ 0 <= a[i], c, d < 998244353
+ 0 <= l < r < N, 0 <= x < N

**方法1:Lazy seg Tree** 

722ms

```c++
struct S {
    mint sum;
    int size;
};
struct F {
    mint c, d;
};
S op(S x, S y) {
    if (x.size == 0) return y;
    if (y.size == 0) return x;
    S s;
    s = S{x.sum + y.sum, x.size + y.size};
    return s;
}
S e() {
    return S{0, 0};
};
S tag(F f, S s) { 
    if (f.c == 1 && f.d == 0) return s;  // 加上这行耗时优化很大
    return S{s.sum * f.c + s.size * f.d, s.size}; 
}
F merge(F x, F y) { return F{x.c * y.c, y.d * x.c + x.d}; }
F id() { return F{1, 0}; }
int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    
    int n, q;
    cin >> n >> q;
    vector<S> a(n, S{0, 1});
    for (int i = 0; i < n; ++i) {
        cin >> a[i].sum;        
    } 
    LazySegTree<S, op, e, F, tag, merge, id> seg(a);
    for (int i = 0, op, l, r, c, d; i < q; ++i) {
        cin >> op;
        if (op == 0) {
            cin >> l >> r >> c >> d;
            seg.apply(l, r, F{c, d});
        } else {
            cin >> l;
            cout << seg.get(l).sum << '\n';
        }
    }

    return 0;
}
```

**对偶线段树**

371ms, 时间和空间效率更高

```c++
template <typename T, typename F, T(*tag)(F, T), F(*merge)(F, F), F(*id)()>
struct CommutativeDualSegmentTree {
    CommutativeDualSegmentTree() {}
    CommutativeDualSegmentTree(vector<T>&& a) : n(a.size()), m((1 << ceil_lg(a.size()))), data(std::move(a)), lazy(m, id()) {}
    CommutativeDualSegmentTree(const std::vector<T>& a) : CommutativeDualSegmentTree(std::vector<T>(a)) {}
    CommutativeDualSegmentTree(int n, const T& fill_value) : CommutativeDualSegmentTree(std::vector<T>(n, fill_value)) {}

    T operator[](int i) const {
        assert(0 <= i and i < n);
        T res = data[i];
        for (i = (i + m) >> 1; i; i >>= 1) res = tag(lazy[i], res);
        return res;
    }
    T get(int i) const { return (*this)[i];}
    void apply(int l, int r, const F& f) {
        assert(0 <= l and r <= n);
        for (l += m, r += m; l < r; l >>= 1, r >>= 1) {
            if (l & 1) apply(l++, f);
            if (r & 1) apply(--r, f);
        }
    }
protected:
    int n, m;
    vector<T> data;
    vector<F> lazy;

    void apply(int k, const F& f) {
        if (k < m) lazy[k] = merge(f, lazy[k]);
        else data[k - m] = tag(f, data[k - m]);
    }
private:
    static int ceil_lg(int x) {   // minimum non-neg x s.t. `n <= 2^x`
        return x <= 1 ? 0 : 32 - __builtin_clz(x - 1);
    }
};
template <typename T, typename F, T(*tag)(F, T), F(*merge)(F, F), F(*id)()>
struct DualSegmentTree : public CommutativeDualSegmentTree<T, F, tag, merge, id> {
    using base_type = CommutativeDualSegmentTree<T, F, tag, merge, id>;
    using base_type::base_type;
    void apply(int l, int r, const F& f) {
        push(l, r);
        base_type::apply(l, r, f);
    }
private:
    void push(int k) {
        base_type::apply(2 * k, this->lazy[k]), base_type::apply(2 * k + 1, this->lazy[k]);
        this->lazy[k] = id();
    }
    void push(int l, int r) {
        static const int log = __builtin_ctz(this->m);
        l += this->m, r += this->m;
        for (int i = log; (l >> i) << i != l; --i) push(l >> i);
        for (int i = log; (r >> i) << i != r; --i) push(r >> i);
    }
};
mint tag(pair<mint, mint> f, mint x) {
    return f.first * x + f.second;
}
pair<mint, mint> merge(pair<mint, mint> f, pair<mint, mint> g) {
    return { f.first * g.first, f.first * g.second + f.second };
}
pair<mint, mint> id() {
    return { 1, 0 };
}
using Segtree = DualSegmentTree<mint, pair<mint, mint>, tag, merge, id>;

int main() {
    std::ios::sync_with_stdio(false); std::cin.tie(nullptr);

    int n, q;
    std::cin >> n >> q;

    std::vector<mint> a(n);
    for (auto &e : a) {
        cin >> e;
    }
    Segtree seg(a);
    for (int i = 0, op, l, r, c, d; i < q; ++i) {
        cin >> op;
        if (op == 0) {
            cin >> l >> r >> c >> d;
            seg.apply(l, r, {c, d});
        } else {
            cin >> l;
            cout << seg.get(l).val() << '\n';
        }
    } 

    return 0;
}
```

### 区间异或查询区间和

[cf242 E](https://codeforces.com/contest/242/problem/E)

一个长度为n的序列，q次操作，每次操作有两种形式
+ 1 l r 输出 a[l,..r]的和
+ 2 l r x 异或操作，对 l <= i <= r 的i，执行 a[i] = a[i] ^ x

+ 1 <= n <= 1e5
+ 1 <= m <= 5e4
+ 0 <= a[i], x <= 1e6

```c++
const int K = 20; // 根据值域调整
struct S{
    array<int,K> a;
    int siz;
};
using F = int;
S op(S l, S r) {
    if(l.siz==0)return r;
    if(r.siz==0)return l;
    S s;
    s.siz = l.siz+r.siz;
    for(int i=0;i<K;++i){
        s.a[i]=l.a[i]+r.a[i];
    }
    return s;
}

S e() { return S{}; }

S tag(F l, S r) {
    if(l==0)return r;
    for(int i=0;i<K;++i){
        if((l>>i)&1)r.a[i]=r.siz-r.a[i];
    }
    return r;
}
F merge(F l, F r) { return l^r; }
F id() { return 0; }

void ac_yyf(int tt) {
    int n, q;
    cin >> n;
    using Seg=LazySegTree<S, op, e, F, tag, merge, id>;
    vector<S> a(n); 
    for (int i = 0, x; i < n; ++i) {
        cin >> x;
        for(int j=0;j<K;++j){
            a[i].a[j]=((x>>j)&1);
        }
        a[i].siz=1;
    }
    Seg seg(a);
    cin >> q;
    for (int i = 0, t, l, r, x; i < q; ++i) {
        cin >> t;
        if (t == 1) {
            cin >> l >> r;
            ll ans = 0;
            auto p = seg.get(l - 1, r).a;
            for (int j = 0; j < K; ++j) {
                ans += p[j] * 1ll * (1 << j);
            }
            cout << ans << '\n';

        } else if (t == 2) {
            cin >> l >> r >> x;
            seg.apply(l - 1, r, x);
        } 
    }
}
```

### 区间翻转求最长连续1数目

[abc 322f](https://atcoder.jp/contests/abc322/tasks/abc322_f)

一个长度为n的01序列，q次查询。
1. 1 l r 将区间[l,r]的1变为0，0变为1
2. 2 l r 输出区间[l,r]中最长连续1的长度

+ 1 <= q <= 1e5
+ 1 <= n <= 5e5

```c++
struct S {
    int pre[2] = {0, 0}; // 前缀0/1长度
    int suf[2] = {0, 0}; // 后缀0/1长度
    int mx[2] = {0, 0};  // 区间连续0/1最大值
    int cnt[2] = {0, 0}; // 区间0/1的数量
};
using F = bool; // 是否取反
S op(S x, S y) {
    S s{};
    for (int i = 0; i < 2; ++i) {
        s.cnt[i] = x.cnt[i] + y.cnt[i];
        s.pre[i] = x.cnt[i ^ 1] ? x.pre[i] : x.cnt[i] + y.pre[i];
        s.suf[i] = y.cnt[i ^ 1] ? y.suf[i] : y.cnt[i] + x.suf[i];
        s.mx[i] = max({x.mx[i], y.mx[i], x.suf[i] + y.pre[i]});
    }
    return s;
}
S e() {
    return S{};
};
S E0() {
    return S{ {1, 0}, {1, 0}, {1, 0}, {1, 0} };
}
S E1() {
    return S{ {0, 1}, {0, 1}, {0, 1}, {0, 1} };
}
S tag(F f, S s) {
    if (!f) return s;
    swap(s.pre[0], s.pre[1]);
    swap(s.cnt[0], s.cnt[1]);
    swap(s.suf[0], s.suf[1]);
    swap(s.mx[0], s.mx[1]);
    return s;
}
F merge(F x, F y) { 
    return x ^ y;
}
F id() { return false; }

void ac_yyf(int tt) {
    int n, q;
    string s;
    cin >> n >> q >> s;
    vector<S> v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = s[i] == '1' ? E1() : E0();
    } 
    LazySegTree<S, op, e, F, tag, merge, id> seg(v);
    for (int i = 0, t, l, r; i < q; ++i) {
        cin >> t >> l >> r;
        if (t == 1) seg.apply(l - 1, r, true);
        else cout << seg.get(l - 1, r).mx[1] << '\n';
    }
}
```

### 区间赋值取反查询最大连续1数目

[luogu p2572](https://www.luogu.com.cn/problem/P2572)

长度为n的01序列，五种操作
+ 0 l r : [l,r]区间内的所有数全变成 0
+ 1 l r : [l,r]区间内的所有数全变成 1
+ 2 l r : [l,r]区间内的所有数全部取反,0变成1，1变成0
+ 3 l r : [l,r]区间内总共有多少个1
+ 4 l r : [l,r]区间内最多有多少个连续的1

+ 1 <= n, m <= 2e5

```c++
struct S {
    int pre[2] = {0, 0}; // 前缀0/1长度
    int suf[2] = {0, 0}; // 后缀0/1长度
    int mx[2] = {0, 0};  // 区间连续0/1最大值
    int cnt[2] = {0, 0}; // 区间0/1的数量
};
using F = array<int, 2>; // (赋值， 取反)
S op(S x, S y) {
    S s{};
    for (int i = 0; i < 2; ++i) {
        s.cnt[i] = x.cnt[i] + y.cnt[i];
        s.pre[i] = x.cnt[i ^ 1] ? x.pre[i] : x.cnt[i] + y.pre[i];
        s.suf[i] = y.cnt[i ^ 1] ? y.suf[i] : y.cnt[i] + x.suf[i];
        s.mx[i] = max({x.mx[i], y.mx[i], x.suf[i] + y.pre[i]});
    }
    return s;
}
S e() {
    return S{};
};
S E0() {
    return S{ {1, 0}, {1, 0}, {1, 0}, {1, 0} };
}
S E1() {
    return S{ {0, 1}, {0, 1}, {0, 1}, {0, 1} };
}
S tag(F f, S s) {
    if (f[0] == -1) {
        if (!f[1]) return s;
        swap(s.pre[0], s.pre[1]);
        swap(s.cnt[0], s.cnt[1]);
        swap(s.suf[0], s.suf[1]);
        swap(s.mx[0], s.mx[1]);
    } else {
        int x = f[0], y = f[0] ^ 1;
        s.pre[x] = s.suf[x] = s.mx[x] = s.cnt[x] = s.cnt[x] + s.cnt[y];;
        s.pre[y] = s.suf[y] = s.mx[y] = s.cnt[y] = 0;
    }
    return s;
}
F merge(F x, F y) { 
    if (x[0] != -1) return F{x[0], 0};
    if (x[1] == 0) {
        return y[0] == -1 ? y : F{y[0], 0};
    } 
    return y[0] == -1 ? F{-1, y[1] ^ 1} : F{y[0] ^ 1, 0};
}
F id() { return {-1, 0}; }

void ac_yyf(int tt) {
    int n, m;
    cin >> n >> m;
    vector<S> a(n);
    for (int i = 0; i < n; ++i) {
        cin >> x;
        a[i] = x == 0 ? E0() : E1();
    }
    LazySegTree<S, op, e, F, tag, merge, id> seg(a);
    for (int i = 0, t, l, r; i < m; ++i) {
        cin >> t >> l >> r;
        r++;
        if (t <= 1) {
            seg.apply(l, r, F{t, 0});
        }
        else if (t == 2) seg.apply(l, r, F{-1, 1});
        else if (t == 3) cout << seg.get(l, r).cnt[1] << '\n';
        else cout << seg.get(l, r).mx[1] << '\n';
    }
}
```

### 区间不同元素数目平方和

[hackerearch](https://www.hackerearth.com/practice/data-structures/advanced-data-structures/segment-trees/practice-problems/algorithm/something-genuine/)

[双周赛116 t4](https://leetcode.cn/problems/subarrays-distinct-element-sum-of-squares-ii/description/)

长度为n的数组a，定义f(l,r)为子数组a[l..r]的不同元素数目，求所有非空子数组f(l,r)的平方和。模1e9+7.

+ 1 <= n <= 2e5
+ 1 <= a[i] <= 2e5

**分析**

设 last[x] 为上一次出现x的坐标，则

```c++
f(l, r) = f(l, r - 1) + 1; // if l > last[nums[r]]
f(l, r) = f(l, r - 1); // if l <= last[nums[r]]
```
设 `g(i) = f(0,0) + f(0, 1) + ... + f(0, i)`

答案即为

```
ans = g(0)^2 + g(1)^2 + ... + g(n-1)^2
```

可以按顺序依次维护g(i)^2。

```c++
struct S {
    mint s1, s2, size;
};
using F = mint;
S op(S x, S y) {
    return S{x.s1 + y.s1, x.s2 + y.s2, x.size + y.size};
}
S e() {
    return S{};
};
S tag(F f, S s) { 
    if (f == 0) return s;
    S res = s;
    res.s2 = s.s2 + s.size * f;
    res.s1 = s.s1 + 2 * f * s.s2 + s.size * f * f;
    return res;
}
F merge(F x, F y) { return x + y; }
F id() { return 0; }
int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    
    int n;
    cin >> n;
    vector<int> a(n);
    for (int i = 0; i < n; ++i) {
        cin >> a[i];
    }

    int mx = *max_element(a.begin(), a.end());
    vector<int> p(mx + 1, -1);

    LazySegTree<S, op, e, F, tag, merge, id> seg(vector<S>(n,{0,0,1}));
    mint ans = 0;
    for (int i = 0; i < n; ++i) {
        seg.apply(p[a[i]] + 1, i + 1, 1);
        p[a[i]] = i;
        ans += seg.get(0, i + 1).s1;
    }
    cout << ans << '\n';

    return 0;
}
```

## 权值线段树

权值线段树维护的是大小在[l, r]的范围，一般配合离散化使用。

### 统计大小在某个范围内的数量

[cses1144](https://vjudge.net/problem/CSES-1144)

一个长度为n的序列，q次操作，每次操作有两种形式
1. `! k x` 将第k个元素修改为x
2. `? a b` 查询元素在[a,b]范围内的数量

+ 1 <= n, q <= 2e5
+ 1 <= a[i], x, a, b <= 1e9
+ 1 <= k <= n

**方法1：平衡树(590ms)**

```c++
#include <bits/stdc++.h>
using namespace std;
#include<ext/pb_ds/assoc_container.hpp>
#include<ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;

template<class T> using ordered_set = tree<T, null_type, less_equal<T>, rb_tree_tag, tree_order_statistics_node_update>;

int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    
    int n,q;
    cin>>n>>q;
    vector<int> a(n);
    ordered_set<int> s;
    for (int i = 0; i < n; ++i) {
        cin>>a[i];  
        s.insert(a[i]);
    }

    for (int i = 0; i < q; ++i) {
        char c;
        cin >> c;
        if (c == '!') {
            int k, x;
            cin >> k >> x;
            k--;
            s.erase(s.find_by_order(s.order_of_key(a[k])));
            a[k] = x;
            s.insert(x);
        } else {
            int l, r;
            cin >> l >> r;

            cout << s.order_of_key(r + 1) - s.order_of_key(l) << '\n';
        }
    }
    return 0;
}
```

**方法2:树状数组(320ms)**

```c++
int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    
    int n, q;
    cin >> n >> q;
    vector<int> a(n);
    for (int i = 0; i < n; ++i) {
        cin >> a[i];    
    }
    vector<array<int, 3>> qs(q);
    char op;
    for (int i = 0; i < q; ++i) {
        cin >> op >> qs[i][1] >> qs[i][2];
        if (op == '!') {
            qs[i][0] = 0;
            a.push_back(qs[i][2]);
        } else {
            qs[i][0] = 1;
            a.push_back(qs[i][1]);
            a.push_back(qs[i][2]);
        }
    }

    Discrete<int> v(a);
    int m = v.size();

    FenwickTree<int> f(m);
    for (int i = 0; i < n; ++i) {
        f.add(v(a[i]), 1);
    }
    for (auto &[op, x, y]: qs) {
        if (op == 0) {
            x--;
            f.add(v(a[x]), -1);
            a[x] = y;
            f.add(v(a[x]), 1);
        } else {
            cout << f.sum(v(x), v(y)) << '\n';
        }
    }

    return 0;
}
```


**方法3:权值线段树(350ms)**

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
    vector<int> a(n);
    for (int i = 0; i < n; ++i) {
        cin >> a[i];    
    }
    vector<array<int, 3>> qs(q);
    char t;
    for (int i = 0; i < q; ++i) {
        cin >> t >> qs[i][1] >> qs[i][2];
        if (t == '!') {
            qs[i][0] = 0;
            a.push_back(qs[i][2]);
        } else {
            qs[i][0] = 1;
            a.push_back(qs[i][1]);
            a.push_back(qs[i][2]);
        }
    }

    Discrete<int> v(a);
    int m = v.size();

    vector<int> p(m);
    for (int i = 0; i < n; ++i) {
        p[v(a[i])]++;
    }

    SegTree<S, op, e> seg(p);

    for (auto &[t, x, y]: qs) {
        if (t == 0) {
            x--;
            seg.set(v(a[x]), seg.get(v(a[x])) - 1);
            a[x] = y;
            seg.set(v(a[x]), seg.get(v(a[x])) + 1);
        } else {
            cout << seg.get(v(x), v(y) + 1) << '\n';
        }
    }
    return 0;
}
```

### 查询集合mex

[cf 817f](https://codeforces.com/contest/817/problem/F)

s初始为空集合，执行n次查询：
+ 1 l r 将[l,r]中所有缺失数字插入集合中
+ 2 l r 将[l,r]中所有存在的数从集合中删除
+ 3 l r 将[l,r]中所有存在的数组删除，插入所有不存在的数字

对于每次查询，输出集合的MEX （MEX>=1)

+ 1 <= n <= 1e5
+ 1 <= l <= r <= 1e18

**1.权值线段树**

将区间离散化，每次查询的MEX值要么是1，要么是某个区间[l,r]的 r+1.
用线段树维护区间中1的数目和0的数目，找到最大的离散化后的下标，满足其0的数目为0.

```c++
struct S {
    int x, y;  // x: 1的数量 y: 0的数量
    S():x(0), y(0){}
    S(int x, int y):x(x), y(y){}
};
S op(S x, S y) {
    S s;
    s = S{x.x + y.x, x.y + y.y};
    return s;
}
S e() {
    return S();
};
using F = int;
S tag(F f, S s) { 
    if (f == 0) return s;
    if (f == 1) return S{s.x + s.y, 0};
    if (f == 2) return S{0, s.x + s.y};
    return S{s.y, s.x}; // 01 翻转
}
// 0:删除标记 1:置1 2:置0 3:翻转
F merge(F x, F y) {
    if (x == 0) return y;
    if (x == 1) return 1;
    if (x == 2) return 2;
    if (y == 0) return 3;
    if (y == 3) return 0;
    if (y == 1) return 2;
    if (y == 2) return 1;
}
F id() { return 0; }

template <class T>
struct Discrete {
    vector<T> xs;
    Discrete(const vector<T>& v) {
        xs = v;
        sort(xs.begin(), xs.end());
        xs.erase(unique(xs.begin(), xs.end()), xs.end());
    }
    int get(const T& x) const {
        return lower_bound(xs.begin(), xs.end(), x) - xs.begin();
    }
    inline int operator()(const T& x) const { return get(x); }
    T operator[](int i) { return xs[i]; }
    int size() const { return xs.size(); }
};

int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    
    int n;
    cin >> n;
    vector<long long> l(n), r(n), a {1};
    vector<int> t(n);
    for (int i = 0; i < n; ++i) {
        cin >> t[i] >> l[i] >> r[i];
        r[i]++;
        a.push_back(l[i]);
        a.push_back(r[i]);
    }

    Discrete<long long> v(a);
    int m = v.size();
    LazySegTree<S, op, e, F, tag, merge, id> seg(vector<S>(m,{0,1}));
    for (int i = 0; i < n; ++i) {
        seg.apply(v(l[i]), v(r[i]), t[i]);
        int r = seg.max_right(0, [&](S s){
            return s.y == 0;
        });
        cout << v[r] << '\n';
    }
    return 0;
}
```