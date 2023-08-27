---
layout: post
title: 倍增算法
date: 2023-07-30
tags: 算法专题  
---


===

Index
---
<!-- TOC -->

- [模板](#模板)
- [跳k次后的位置](#跳k次后的位置)
- [城市之间最少边数](#城市之间最少边数)
- [两点之间的最少天数](#两点之间的最少天数)
- [树上倍增](#树上倍增)
  - [](#)


<!-- /TOC -->

**模板**

```c++
template <class S, S (*op)(S, S)> class BiLiftring {
    int n = 0;
    vector<vector<int>> _nexts;
    vector<vector<S>> _prods;

    void build_next() {
        vector<int> t(n);
        vector<S> p(n);

        for (int i = 0; i < n; ++i) {
            if (int j = _nexts.back()[i]; isin(j)) {
                t[i] = _nexts.back()[j], p[i] = op(_prods.back()[i], _prods.back()[j]);
            } else t[i] = j, p[i] = _prods.back()[i];
        }
        _nexts.emplace_back(move(t));
        _prods.emplace_back(move(p));
    }

    inline bool isin(int i) const noexcept { return 0 <= i and i < n; }

public:
    // (up to) 2^d steps from `s`
    // Complexity: O(d) (Already precalculated) / O(nd) (First time)
    int pow_next(int s, int d) {
        assert(isin(s));
        while (int(_nexts.size()) <= d) build_next();
        return _nexts.at(d).at(s);
    }

    // Product of (up to) 2^d elements from `s`
    const S &pow_prod(int s, int d) {
        assert(isin(s));
        while (int(_nexts.size()) <= d) build_next();
        return _prods.at(d).at(s);
    }

    BiLiftring() = default;
    BiLiftring(const vector<int> &g, const vector<S> &w)
        : n(g.size()), _nexts(1, g), _prods(1, w) {
        assert(g.size() == w.size());
    }

    // (up to) k steps from `s`
    template <class Int> int kth_next(int s, Int k) {
        for (int d = 0; k > 0 and isin(s); ++d, k >>= 1) {
            if (k & 1) s = pow_next(s, d);
        }
        return s;
    }

    // Product of (up to) `len` elements from `s`
    template <class Int> S get(int s, Int len) {
        assert(isin(s)); assert(len > 0);
        int d = 0;
        while (!(len & 1)) ++d, len /= 2;

        S ret = pow_prod(s, d);
        s = pow_next(s, d);
        for (++d, len /= 2; len and isin(s); ++d, len /= 2) {
            if (len & 1) ret = op(ret, pow_prod(s, d)), s = pow_next(s, d);
        }
        return ret;
    }

    // `start` から出発して「`left_goal` 以下または `right_goal` 以上」に到達するまでのステップ数
    // 単調性が必要
    int dis_mono(int start, int left_goal, int right_goal) {
        assert(isin(start));

        if (start <= left_goal or right_goal <= start) return 0;

        int d = 0;
        while (left_goal < pow_next(start, d) and pow_next(start, d) < right_goal) {
            if ((1 << d) >= n) return -1; ++d;
        }
        int ret = 0, cur = start;
        for (--d; d >= 0; --d) {
            if (int nxt = pow_next(cur, d); left_goal < nxt and nxt < right_goal) {
                ret += 1 << d, cur = nxt;
            }
        }
        return ret + 1;
    }

    template <class F> long long max_len(const int s, F f, const int maxd = 60) {
        assert(isin(s));
        int d = 0;
        while (d <= maxd and f(pow_prod(s, d))) {
            if (!isin(pow_next(s, d))) return 1LL << maxd; ++d;
        }
        if (d > maxd) return 1LL << maxd;
        --d;
        int cur = pow_next(s, d);
        long long len = 1LL << d;
        S p = pow_prod(s, d);
        for (int e = d - 1; e >= 0; --e) {
            if (S nextp = op(p, pow_prod(cur, e)); f(nextp)) {
                swap(p, nextp);
                cur = pow_next(cur, e);
                len += 1LL << e;
            }
        }
        return len;
    }
};
```

**模板2**

```c++
struct BiLifting {
    int N, INVALID, lgD;
    vector<vector<int>> mat;
    BiLifting() : N(0), lgD(0) {}
    BiLifting(const vector<int> &vec_nxt, int INVALID = -1, int lgd = 0)
        : N(vec_nxt.size()), INVALID(INVALID), lgD(lgd) {
        while ((1LL << lgD) < N) lgD++;
        mat.assign(lgD, vector<int>(N, INVALID));
        mat[0] = vec_nxt;
        for (int i = 0; i < N; i++)
            if (mat[0][i] < 0 or mat[0][i] >= N) mat[0][i] = INVALID;
        for (int d = 0; d < lgD - 1; d++) {
            for (int i = 0; i < N; i++)
                if (mat[d][i] != INVALID) mat[d + 1][i] = mat[d][mat[d][i]];
        }
    }
    int kth_next(int now, long long k) {
        if (k >= (1LL << lgD)) exit(8);
        for (int d = 0; k and now != INVALID; d++, k >>= 1)
            if (k & 1) now = mat[d][now];
        return now;
    }

    // Distance from l to [r, infty)
    // Requirement: mat[0][i] > i for all i (monotone increasing)
    int distance(int l, int r) {
        if (l >= r) return 0;
        int ret = 0;
        for (int d = lgD - 1; d >= 0; d--) {
            if (mat[d][l] < r and mat[d][l] != INVALID) ret += 1 << d, l = mat[d][l];
        }
        if (mat[0][l] == INVALID or mat[0][l] >= r)
            return ret + 1;
        else
            return -1; // Unable to reach
    }
};
```

### 跳k次后的位置

[yukicoder 1013](https://yukicoder.me/problems/no/1013)

给一个1-n的排列，循环无数次。
当处于第i个位置时，设第i个位置的数字是x, 会向右跳x个位置，
输入整数k，求当从第1，2,n个位置出发时，向右跳k次后所到达的位置。

+ 1 <= n <= 1e5
+ 1 <= k <= 1e9

**分析**

从 i 到 (i+a[i]) % n 连边，倍增维护求i跳2^i次后的位置

```c++
using S = long long;
S op(S l, S r) { return l + r; }

int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    
    int n, k;
    cin >> n >> k;
    vector<long long> a(n);
    vector<int> g(n); // 图结构
    for (int i = 0; i < n; ++i) {
      cin >> a[i];
        // 从 i 到 (i+a[i]) % n 连边
      g[i] = (i + a[i]) % n;
    }
    BiLiftring<S, op> b(g, a);
    for (int i = 0; i < n; ++i)
      cout << i + 1 + b.get(i, k) << '\n';
   
    return 0;
}
```

### 城市之间最少边数

[yukicoder 2242](https://yukicoder.me/problems/no/2242)

n个城市，高度分别为h[1],..h[n]. 另给一个长度为n的数组t，从城市i出发，可以到达高度不超过t[i]的所有其他城市，
有q次询问，每次询问给出x,y,求从x到y的最少边数，如果无法到达y，输出-1.

+ 2 <= n <= 2e5
+ 1 <= q <= 2e5
+ 1 <= h[i], t[i] <= 1e9
+ 1 <= a[i], b[i] <= n
+ a[i] != b[i]

```c++
// 模板2
// Discrete
int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    
    int n;
    cin >> n;
    vector<int> h(n), t(n);
    for (int &x : h)
        cin >> x;
    for (int &x : t)
        cin >> x;

    auto z = h;
    z.insert(z.end(), t.begin(), t.end());
    Discrete<int> v(z);

    for (auto &x : h) 
        x = v(x);
    for (auto &x : t) 
        x = v(x);
    int m = v.size();
    vector<int> g(m);
    iota(g.begin(), g.end(), 0);
    for (int i = 0; i < n; ++i) 
        g[h[i]] = max(g[h[i]], t[i]);
    for (int i = 1; i < m; ++i)
        g[i] = max(g[i], g[i - 1]);

    BiLifting bl(g);
    int q;
    cin >> q;
    for (int i = 0; i < q; ++i) {
        int x, y;
        cin >> x >> y;
        x--, y--;
        int l = t[x], r = h[y];
        auto c = bl.distance(l, r);
        if (c >= 0) c++;
        cout << c << '\n';
    }
   
    return 0;
}
```

### 两点之间的最少天数

[arc060 E](https://atcoder.jp/contests/arc060/tasks/arc060_c)

一条直线上有n个旅馆，第i个旅店坐标为xi, 旅行者有两个原则：
1. 一天行走的距离不超过l
2. 每天的起点和终点必须是某个旅店的位置。
q次询问，每次询问给定x，y，求从第x个旅店到第y个旅店所需要的最短天数，输入保证一定能从x到y。

+ 2 <= n <= 1e5
+ 1 <= x1 < x2 ... < xn <= 1e9
+ 1 <= l <= 1e9
+ xi - x(i-1) <= l
+ 1 <= x, y <= n
+ x != y

**分析**

r[i]: 表示从第i个节点1天能到达的最右边节点。可以用双指针求出数组r。
倍增维护从第i个节点经过2^i天能够到达的节点。

```c++
int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    
    int n;
    cin >> n;
    vector<int> X(n);
    for (int &x : X)
        cin >> x;

    int L, Q;
    cin >> L >> Q;
    vector<int> r(n);
    for (int i = 0, j = 0; i < n; ++i) {
        while (j < n - 1 && X[j + 1] <= X[i] + L) j++;
        r[i] = j;    
    }

    BiLifting d(r);

    for (int i = 0; i < Q; ++i) {
        int x, y;
        cin >> x >> y;
        x--, y--;
        if (x > y) swap(x, y);
        cout << d.distance(x, y) << '\n';
    }
    
    return 0;
}
```
