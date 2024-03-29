---
layout: post
title: 树上启发式合并&欧拉序
date: 2023-08-10
tags: 算法专题  
---


===

Index
---
<!-- TOC -->

- [启发式合并](#启发式合并模板)
  - [子树出现次数最多的颜色编号之和](#子树出现次数最多的颜色编号之和)
  - [树上数颜色](#树上数颜色)
- [欧拉序](#欧拉序模板)
  - [查询路径和](#查询路径和)
  - [路径上等于k的节点数](#路径上等于k的节点数)
  - [路径上被k整除的节点数](#路径上被k整除的节点数)
- [bfs序](#bfs序)
  - [权值大于p的节点最小高度](#权值大于p的节点最小高度)



<!-- /TOC -->

树上启发式合并用于解决树上对所有子树的查询问题，一般不带修改，总时间复杂度为 O(nlog(n))

## 启发式合并模板

```c++
struct DsuOnTree {
    int n, lst_rt;
    vector<vector<int>> g;
    vector<int> siz, ver, in, out;
    DsuOnTree(int n_ = 0) : n(n_), lst_rt(-1), g(n_), in(n_), out(n_), siz(n_, 1){}

    void add_edge(int u, int v, bool bi_dre = true) {
        g[u].push_back(v);
        if(bi_dre) g[v].push_back(u);
    }

    void dfs1(int u, int p) {
        if (p != -1) g[u].erase(find(g[u].begin(), g[u].end(), p));
        in[u] = ver.size(), ver.push_back(u);
        for (int& v : g[u]) if (v != p) {
            dfs1(v, u);
            siz[u] += siz[v];
            if (siz[v] > siz[g[u][0]]) {
                swap(v, g[u][0]);  // g[u][0] 存储u节点的重儿子
            }
        }
        out[u] = ver.size();
    }

    template <class F1, class F2, class F3>
    void build(int root, F1& Add, F2& Del, F3& Calc) {
        dfs1(root, -1);

        auto dfs = [&](auto &dfs, int u, bool keep) -> void {
            int son = g[u].size() ? g[u][0] : -1;
            for (int i = 1; i < g[u].size(); ++i) 
                dfs(dfs, g[u][i], false);
            
            if (son != -1) dfs(dfs, son, true);
            for (int i = 1; i < g[u].size(); ++i) 
                for (int j = in[g[u][i]]; j < out[g[u][i]]; ++j) 
                    Add(ver[j]);
            Add(u);
            Calc(u);
            if (!keep) for (int i = in[u]; i < out[u]; ++i) Del(ver[i]);
        };

        dfs(dfs, 0, false);
    }
};
DsuOnTree g(n);
auto Add = [&](int v) {
    ;
};

auto Del = [&](int v) {
    ;
};

auto Calc = [&](int v) {
    ;
};

g.build(0, Add, Del, Calc);
```

### 子树出现次数最多的颜色编号之和

[cf600E](https://codeforces.com/contest/600/problem/E)

n个节点的树，根节点为1，每个节点的颜色为c[i]。 如果一个颜色再以x为根的子树中出现次数最多，称其在以x为根的子树中占主导地位，一个子树可以有多个颜色占主导地位。
求每个节点i，求以i为根的子树中占主导地位的颜色的编号和。

+ 1 <= c[i] <= n <= 1e5

[submission](https://codeforces.com/contest/600/submission/218058771)

```c++
int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    
    int n;
    cin >> n;
    vector<int> a(n);
    for (int &x : a) 
      cin >> x;

    vector<int> cnt(n + 1);
    vector<long long> res(n);
    int mx = 0;
    long long sum = 0;
    vector<int> his;

    DsuOnTree g(n);
 
    for (int i = 1; i < n; ++i) {
      int u, v;
      cin >> u >> v;
      u--, v--;
      g.add_edge(u, v);
    }

    auto Add = [&](int v) {
        int x = a[v];
        cnt[x]++;
        his.push_back(x);
        if (mx < cnt[x]) {
          mx = cnt[x], sum = x;
        } else if (mx == cnt[x]) sum += x;
    };

    auto Del = [&](int v) {
        for (auto &&x : his){
          cnt[x] = 0;
        }
        his = {};
        mx = sum = 0;
    };

    auto Calc = [&](int v) {
        res[v] = sum;
    };

    g.build(0, Add, Del, Calc);

    for (int i = 0; i < n; ++i) {
        cout << res[i] << " \n"[i == n - 1];  
    }
    return 0;
}
```

### 树上数颜色

[luogu U41492](https://www.luogu.com.cn/problem/U41492)

一颗n个节点的树，根为1，每个节点颜色为c[i], m个询问，每个询问给定x，输出以x为根的子树包含的颜色种类。

+ 1 <= m, c[i] <= n <= 1e5

```c++
int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    
    int n, m;
    cin >> n;
  
    DsuOnTree g(n);
    for (int i = 1; i < n; ++i) {
      int u, v;
      cin >> u >> v;
      u--, v--;
      g.add_edge(u, v);
    }

    vector<int> a(n);
    for (int &x : a) 
        cin >> x;

    set<int> s;
    vector<int> ans(n);
    auto Add = [&](int v) {
        int x = a[v];
        s.insert(x);
    };
    auto Del = [&](int v) {
        s.clear();
    };
    auto Calc = [&](int v) {
        ans[v] = s.size();
    };
    g.build(0, Add, Del, Calc);

    cin >> m;
    for (int i = 0, x; i < m; ++i) {
        cin >> x;    
        cout << ans[x -  1] << '\n';
    }
    return 0;
}
```

## 欧拉序模板

```c++
struct EulerTour {
  int n, m;
    vector<int> in, out, dep, seg;
    template <typename G>
    EulerTour(const G g, int n, int root = 0) : n(n), m(ceil_pow(2 * n)), in(n), out(n), dep(n + 1) {
        seg.assign(2 * m, n);
        dfs(g, root);
        for (int k = m - 1; k > 0; --k) seg[k] = argmin(seg[(k << 1) | 0], seg[(k << 1) | 1]);
    }
    template <typename E, typename EdgeToNode>
    EulerTour(const vector<vector<E>> &g, const EdgeToNode e2n, int root = 0) :
        EulerTour([&](int u, auto f) { for (const E &e : g[u]) f(e2n(e)); }, g.size(), root) {}
    EulerTour(const vector<vector<int>> &g, int root = 0) :
        EulerTour([&](int u, auto f) { for (int v : g[u]) f(v); }, g.size(), root) {}
    int lca(int u, int v) const {
        if (in[u] > in[v]) return lca(v, u);
        int res = n;
        for (int l = m + in[u], r = m + in[v] + 1; l < r; l >>= 1, r >>= 1) {
            if (l & 1) res = argmin(res, seg[l++]);
            if (r & 1) res = argmin(res, seg[--r]);
        }
        return res;
    }
    inline int dis(int u, int v) const { return dep[u] + dep[v] - 2 * dep[lca(u, v)]; }
    inline int parent(int u) const {
        int p = seg[m + out[u]];
        return p == n ? -1 : p;
    }
    template <typename F>
    void node_query(int u, int v, F &&f) {
      int l = lca(u, v);
      f(in[l], in[u]);
      f(in[l] + 1, in[v]);
    }
    template <typename F>
    void edge_query(int u, int v, F &&f) {
      int l = lca(u, v);
      f(in[l] + 1, in[u]);
      f(in[l] + 1, in[v]);
    }
    template <typename G>
    void dfs(G g, int root) {
        dep[root] = 0, dep[n] = numeric_limits<int>::max();
        int k = 0;
        auto f = [&](auto self, int u, int p) -> void {
            in[u] = k, seg[m + k++] = u;
            g(u, [&](int v){ if (v != p) dep[v] = dep[u] + 1, self(self, v, u); });
            out[u] = k, seg[m + k++] = p;
        };
        f(f, root, n);
    }
    inline int argmin(int u, int v) const { return dep[u] < dep[v] ? u : v; }
    static int ceil_pow(const int n) {
        return 1 << (n <= 1 ? 0 : 32 - __builtin_clz(n - 1));
    }
};
```

### 查询路径和

[vertex_add_path_sum](https://judge.yosupo.jp/problem/vertex_add_path_sum)

一颗n个节点的树，每个节点有个值a[i], q个操作
1. 0 p x 赋值 a[p] = a[p] + x
2. 1 u v 输出从节点u到节点v路径上的权值和

+ 1 <= n,q <= 5e5
+ 0 <= a[i], x <= 1e9
+ 0 <= p, u, v < n

```c++
// EulerTour
// FenwickTree
int main() {
    
    int n, q;
    cin >> n >> q;
    vector<int> a(n);
    for (int i = 0; i < n; ++i) {
        cin >> a[i];    
    }

    vector<vector<int>> g(n);
    for (int i = 1, u, v; i < n; ++i) {
        cin >> u >> v;
        g[u].push_back(v);
        g[v].push_back(u);
    }

    EulerTour et(g);
    FenwickTree<long long> f(2 * n);
    for (int i = 0; i < n; ++i) {
        f.set(et.in[i], a[i]);
        f.set(et.out[i],-a[i]);
    }

    for (int i = 0, t; i < q; ++i) {
        cin >> t;
        if (t == 0) {
            int p, x;
            cin >> p >> x;
            f.add(et.in[p], x);
            f.add(et.out[p], -x);
        } else {
            int u, v;
            cin >> u >> v;
            long long s = 0;
            et.node_query(u, v, [&](int x, int y){
                s += f.sum(x, y);
            });
            cout << s << '\n';
        }
    }

    return 0;
}
```

### 路径上等于k的节点数

[hackerearth cir_9_e](https://www.hackerearth.com/problem/algorithm/workers-1faf1dc4/)

一颗n个节点的树，每个节点有个值a[i], q个询问，每次询问给定 u,v,k,求从u到v路径上值等于k的节点数。

+ 1 <= n, q <= 1e5
+ 0 < a[i], k <= 1e9

**分析**

离线查询，对于每个k单独查询，使用欧拉序维护树上前缀和。

```c++
// EulerTour
// FenwickTree

vector<int> path_equal_k(vector<int> &a, vector<vector<int>> &es, vector<vector<int>> &qs){
    int n=sz(a),q=sz(qs);
    vector<vector<int>> g(n);
    for(auto&e:es){
        g[e[0]].push_back(e[1]);
        g[e[1]].push_back(e[0]);
    }
    EulerTour e(g);
    vector<int> v=a;
    for(auto&q1:qs)
        v.push_back(q1[2]);
    Discrete<int> d(v);
    int m=sz(v);

    vector<vector<int>> c(m), p(m);
    f0(n)c[d(a[i])].push_back(i);
    f0(q)p[d(qs[i][2])].push_back(i);

    FenwickTree<int> f(n*2);
    auto add=[&](int k, int v){
        for (auto &x: c[k]) {
            f.add(e.in[x], v);
            f.add(e.out[x], -v);
        }
    };
    vector<int> ans(q);

    f0(m) {
        if(sz(c[i])==0||sz(p[i])==0) continue;
        add(i, 1);
        for(auto&j:p[i]){
            int s=0;
            e.node_query(qs[j][0],qs[j][1],[&](int x, int y){
                s+=f.sum(x,y);
            });
            ans[j]=s;
        }
        add(i, -1);
    }
    return ans;
}
```

### 路径上被k整除的节点数

[hackerearth cir11](https://www.hackerearth.com/problem/algorithm/treepath-divisibility-621877f1/)

一棵n个节点的树，每个节点有个权值，n个节点的权值形成1-n的排列，q次询问，每次询问给定u,v,k，求从u到v的路径上有多少个节点的权值能被k整除。

+ 2 <= n, q <= 1e5
+ 1 <= u, v, k <= n

```c++
void ac_yyf(int tt) {
    int n, q;
    cin >> n >> q;
    vector<int> p(n + 1);
    for (int i = 0; i < n; ++i) {
        cin >> x;
        p[x] = i;    
    }
    vector<vector<int>> g(n);
    for (int i = 0, u, v; i < n - 1; ++i) {
        cin >> u >> v;
        u--, v--;
        g[u].push_back(v);
        g[v].push_back(u);
    }
    EulerTour e(g);

    vector<vector<array<int, 3>> > qs(n + 1);
    for (int i = 0; i < q; ++i) {
        int u, v, k;
        cin >> u >> v >> k;
        u--, v--;
        qs[k].push_back({u, v, i});
    }

    vector<int> ans(q);
    FenwickTree<int> f(2 * n);

    for (int i = 1; i <= n; ++i) {
        if (qs[i].size() == 0) continue;
        for (int k = i; k <= n; k += i) {
            f.add(e.in[p[k]], 1);
            f.add(e.out[p[k]], -1);
        }
        for (auto &[u, v, idx] : qs[i]) {
            int s = 0;
            e.node_query(u, v, [&](int x, int y){
                s += f.sum(x, y);
            });
            ans[idx] = s;
        }
        for (int k = i; k <= n; k += i) {
            f.add(e.in[p[k]], -1);
            f.add(e.out[p[k]], 1);
        }

    }

    for (int i = 0; i < q; ++i) {
        cout << ans[i] << "\n";    
    }
}
```

## bfs序

bfs序可以用来处理按照节点深度查询的问题，bfs序能够将深度在区间[l,r] 内的节点编号为一个连续的区间，可以借助线段树进行查询和修改。

**模板**

```c++
struct BFSTree {
    int n;
    bool is_build;
    vector<vector<int>> g;
    vector<int> dep, seq, ids, h; //h[i]:高度为i的节点起始编号
    BFSTree(){init(0);}
    BFSTree(int _n){init(_n);};
    void init(int _n) {
        this->n = _n;
        is_build = false; h.reserve(n);
        ids.resize(n); seq.resize(n);
        dep.resize(n); g.resize(n);
    }
    void add_edge(int u, int v) {
        g[u].push_back(v);
        g[v].push_back(u);
    }
    void build(int root = 0) {
        is_build = true;
        queue<int> q;
        vector<bool> vis(n);
        q.push(root);
        vis[root] = true;
        int t = 0, d = 0;
        while (q.size()) {
            int m = q.size();
            for (int i = 0; i < m; ++i) {
                auto u = q.front(); q.pop();
                ids[u] = t++, dep[u] = d;
                seq[ids[u]] = u;
                for (int v : g[u]) if (!vis[v]) {
                    q.push(v); vis[v] = true;
                }
            }
            h.push_back(t - 1);
            d++;
        }
    }

    int new_dep(int u) const { assert(is_build); return dep[seq[u]];}
    int ori_dep(int u) const { assert(is_build); return dep[u];}

    template<typename F> // 处理高度从[lo, hi]的区间
    void depth(int lo, int hi, F &&f) {
        if (lo > hi) swap(lo, hi);
        int m = h.size();
        if (hi >= m) hi = m - 1;
        if (lo > hi) return;
        int r = hi == m - 1 ? n : h[hi + 1];
        f(h[lo], r); // 左闭右开区间
    }
};
```

### 权值大于p的节点最小高度

[hackerearth tree_querys](https://www.hackerearth.com/problem/algorithm/tree-queries-4-044903fe/?source=list_view)

n个节点的树，每个节点有值a[i], 根节点为1，q次询问，每次询问有两种形式
+ 1 u x 将节点u处的值增加x
+ 2 p 查询离根节点距离最小的节点的距离，满足节点的权值严格大于p

+ 2 <= n, q <= 1e5
+ 1 <= a[i], x <= 1e6
+ 1 <= p <= 1e12

```c++
using S = long long;
S op(S x, S y) {
   return max(x,y);
}
S e() {
    return S{};
}
vector<int> tree_querys(vector<int> &a, vector<vector<int>> &es, vector<vector<ll>> &qs){
    int n = a.size();
    BFSTree g(n);  // bfs序，将每次查询转换为一个连续区间的查询
    for(auto &e : es){
        g.add_edge(e[0], e[1]);
    }
    g.build();
    SegTree<S, op, e> seg(n); // 线段树，进行修改和二分查询
    for (int i = 0; i < n; ++i) {
        seg.set(g.ids[i], a[i]);
    }
    vector<int> ans;
    for(auto &q : qs){
        if (q[0] == 1) {
            seg.set(g.ids[q[1]], seg.get(g.ids[q[1]]) + q[2]);
        }  else {
            int r = seg.max_right(0,[&](S x){
                return x <= q[1];
            });
            int s = r >= n ? -1 : g.new_dep(r);
            ans.push_back(s);
        }
    }
    return ans;
}
```