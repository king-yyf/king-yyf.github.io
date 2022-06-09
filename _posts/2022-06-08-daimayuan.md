---
layout: post
title: 代码源每日一题 div1
date: 2022-06-08
tags: 算法专题  
---


===

Index
---
<!-- TOC -->

- [div1](#div1)
  - [子串的最大差](#子串的最大差)
  - [区间中不大于x的数的数目](#区间中不大于x的数的数目)
  - [树上路径异或和](#树上路径异或和)
  - [最小或值生成树](#最小或值生成树)
  - [统计子数组和模k等于子数组长度的数量](#统计子数组的数量)
- [div2](#div2)
  
   
<!-- /TOC -->


## div1

### 子串的最大差

定义序列的最大差为序列中最大数与最小数的差。比如 (3,1,4,5,6) 的最大差为 6−1=5 , (2,2) 的最大差为 2−2=0 。

定义一个序列的子串为该序列中连续的一段序列。

给定一个长度为 𝑛 的数组 𝑎1,𝑎2,…,𝑎𝑛，请求出这个序列的所有子串的最大差之和。

+ 1 ≤ 𝑛 ≤ 500000
+ 0 ≤ 𝑎𝑖 ≤ 1e8

**分析**

子序列的最大数和最小数相互独立，分别计算序列的所有子串的最大值和和最小值和，再求差。

```c++
long long calc(vector<int> &a){
    int n = a.size();
    auto get = [&](function<bool(int,int)> f) -> long long {
        vector<int> left(n, -1),right(n, n);
        stack<int> sk;
        for (int i = 0; i < n; ++i) {
            while (!sk.empty() && f(a[sk.top()],a[i])) {
                right[sk.top()] = i;
                sk.pop();
            }
            if (!sk.empty()) left[i] = sk.top();
            sk.push(i);
        }
        long long s = 0;
        for (int i = 0; i < n; ++i) {
            s += 1ll * (i - left[i]) * (right[i] - i) * a[i];
        }
        return s;
    };

    return get([&](int x,int y){return x < y;}) - get([&](int x,int y){return x > y;});
}
```

## 区间中不大于x的数的数目

在给定 𝑁 长的数组 {𝐴} 中进行 𝑄 次询问 [𝐿𝑖,𝑅𝑖] 区间中不大于 𝐻𝑖 的元素个数。

+ 1 <= n <= 100000
+ 1 <= ai <= 1e9

**分析**

我们用树状数组维护区间和，将数组 a 带着下标，按照值大小从小到大排序，再将每个询问，按照 h 值从小到大排序。然后我们遍历每个询问，对于当前的 h，将数组 a 中小于等于的 h 的数的下标 pos 进行 add(pos,1), 这个询问的答案就是 ask(r)-ask(l-1);

```c++
vector<int> calc(vector<int>& a, vector<vector<int>>& qs) {
    int n = a.size(), m = qs.size();
    vector<int> ans(m), tr(n);

    auto add = [&](int c, int x) {
        for (; c <= n; c += c & -c) tr[c - 1] += x;
    };

    auto ask = [&](int c) {
        int s = 0;
        for (; c; c -= c & -c) s += tr[c - 1];
        return s;
    };

    vector<array<int,2>> v;
    vector<array<int,4>> q;
    for (int i = 0; i < n; ++i) 
        v.push_back({a[i], i + 1});
    for (int i = 0; i < m; ++i) 
        q.push_back({qs[i][0], qs[i][1], qs[i][2], i});
    sort(v.begin(), v.end(), [&](array<int,2> x, array<int,2> y){return x[0] < y[0];});
    sort(q.begin(), q.end(), [&](array<int,4> x, array<int,4> y){return x[2] < y[2];});

    for (int i = 0, j = 0; i < m; ++i) {
        while (j < n && v[j][0] <= q[i][2]) 
            add(v[j++][1], 1);
        ans[q[i][3]] = ask(q[i][1]) - ask(q[i][0] - 1);
    }
    return ans;
}
```

### 树上路径异或和

给出 𝑛 个点的一棵树，每个点有各自的点权，多次询问两个点简单路径所构成点集的异或和。

**输入格式**
第一行两个数字 𝑛 和 𝑚 , 𝑛 表示点数，𝑚 表示询问次数 。

接下来一行 𝑛 个整数 𝑎1,𝑎2,…,𝑎𝑛 ，表示每个点的点权。

接下来 𝑛−1 行 , 每行两个整数 𝑢,𝑣 ，表示点 𝑢 和点 𝑣 之间存在一条边。

再接下来 𝑚 行，每行两个整数 𝑢,𝑣 ，表示询问点 𝑢 到点 𝑣 的简单路径所构成点集的异或和。

**输出格式**
输出 𝑚 行，对于每个询问，输出一行。


+ 1 <= n,m <= 200000
+ 1 <= ai <= 1e6

**分析**

现求出每个点到跟节点的路径异或值，则两个点的路径异或值等于两个节点到根节点的异或值异或上最近公共祖先。

```c++
void solve(){
    cin >> n >> m;
    vector<int> a(n),sum(n);
    rd(a);
    LCA g(n);

    for(int i = 0; i < n-1; ++i){
        cin>>x>>y;
        x--,y--;
        g.add_edge(x,y);
    }

    function<void(int,int)> dfs=[&](int u, int fa){
        if(fa==-1) sum[u]=a[u];
        else sum[u]=sum[fa]^a[u];
        for (auto v : g.g[u]) {
            if (v != fa) {
                dfs(v, u);
            }
        }
    };
    dfs(0,-1);
    g.complete();

    while(m--){
        cin>>x>>y;
        x--,y--;
        int u=g.lca(x,y);
        int t=sum[x]^sum[y]^a[u];
        cout<<t<<"\n";
    }
}
```


### 最小或值生成树

给出𝑛个点， 𝑚条边的无向图， 每条边连接𝑢,𝑣两个端点，边权为𝑤， 求图的生成树的最小代价。

在这道题中， 我们定义一棵生成树的代价为他所有边的边权按位或得到的值。

+ 1 <= u,v <= 2e5
+ n - 1 <= m <= 4e5
+ 1 <= w <= 1e9

**分析**

贪心 + 并查集

从高位到低位，判断某一位能否取0,使用并查集判断是否是生成树。

假设我们前面确定了这些位构成的数位res，且当前是第k 位，我们假设当前位取0 ，那么什么样的边会会在我们的答案中呢，首先这个边边权v的第k位一定是0 ，否则一或会使这一位位1，由于后面的位不确定因此后面取什么都行，但是前面一定要符合我们前面的res即这条边或完res，第k位之前的位置和res中应该相同，不同的话代表这条边在前面贪心的时候就去除了，要判段能否生成树，直接判断最后这个图是否连通即可。


```c++
const int N = 2e5 + 5;

int f[N];
int find(int x) {
    return x == f[x] ? x : f[x] = find(f[x]);
}

int n,m,x,y,k,q,res;
void solve(){
    cin>>n>>m;
    vector<array<int, 3>> a(m);
    for(int i=0;i<m;++i){
        cin>>a[i][0]>>a[i][1]>>a[i][2];
        a[i]={x,y,k};
    }
    
    auto chk = [&](int k)->bool{
        iota(f+1,f+n+1,1);
        int x=res,cnt = 0;
        for(int i=0;i<m;++i){
            int t=(1<<k)-1;
            t&=a[i][2];
            t|=x;
            if((a[i][2]>>k&1)==0&&t==(x|a[i][2])) {
                f[find(a[i][0])]=find(a[i][1]);
            }
        }
        for (int i=1;i<=n;++i){
            if(i==f[i]) cnt++;
        }
        return cnt!=1;
    };
 
    for(int i=30;~i;--i){
        if(chk(i)) res |= (1 << i);
    }
    cout << res << "\n";
}
```

### 统计子数组的数量

给定长度为n的数组a和k, 如果一个子数组和除以k的余数正好等于子数组长度,则称其为好子数组，统计好子数组的数量。

+ 1 <= n <= 2e5
+ 1 <= k, a[i] <= 1e9

**分析**

+ 显然子数组长度小于k，因为模k的范围在[0,k-1]

根据条件：

```
 (a[i] + a[i+1] + ,,, + a[j]) % k = j - i + 1
```

设 s 为 a 的前缀和数组，则上式变为:
    
```
(s[j] - s[i - 1]) % k = j - i + 1
```

即：

```
(s[j] - j) % k = (s[i - 1] - (i - 1)) % k
```

使用map统计即可，注意子数组长度要小于k。

```c++
long long countSubarrays(vector<int> &a, int k) {
    int n = a.size();
    vector<int> s(n + 1);
    map<int, int> cnt;

    for (int i = 0; i < n; ++i) {
        s[i + 1] = (s[i] + a[i]) % k;
    }
    long long ans = 0;
    for (int i = 0; i <= n; ++i) {
        if (i >= k) cnt[((s[i-k] - i)%k + k)%k]--;
        ans += cnt[((s[i] - i)%k + k)%k];
        cnt[((s[i] - i)%k + k)%k]++;
    }
    return ans;
}
```

## div2