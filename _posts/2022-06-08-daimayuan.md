---
layout: post
title: 杂题选讲
date: 2022-06-08
tags: 算法专题  
---


===

Index
---
<!-- TOC -->

- [daimayuan](#daimayuan)
  - [子串的最大差](#子串的最大差)
  - [区间中不大于x的数的数目](#区间中不大于x的数的数目)
  - [树上路径异或和](#树上路径异或和)
  - [最小或值生成树](#最小或值生成树)
  - [统计子数组和模k等于子数组长度的数量](#统计子数组的数量)
  - [异或后最少逆序对数](#异或后最少逆序对数)
  - [方块消失的操作次数](#方块消失的操作次数)
  - [工作安排](#工作安排)
  - [树上三角形数](#树上三角形数)
  - [环上分段和的最大公约数](#环上分段和的最大公约数)
  - [字典序最小](#字典序最小)
  - [好序列](#好序列)
  - [区间和](#区间和)
- [acwing/牛客](#acwing)
  - [平均值大于k的最长子数组长度](#平均值大于k的最长子数组长度)
  - [所有子数组平均数之和](#所有子数组平均数之和)
  - [均值大于等于k的子数组数目](#均值大于等于k的子数组数目)
  - [数对](#数对)
  
   
<!-- /TOC -->


## daimayuan

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

### 区间中不大于x的数的数目

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

**思路2**

可以统计该路径上每一位有多少个1，如果有奇数个，则异或和为1，否则为0，这种方法还可以扩展到一并解决路径与，路径或问题，例如对于或问题，如果路径上该位存在1，则结果的该位为1，否则为0，对于与问题，路径上该位1的个数等于路径上元素个数，则该位为1，否则为0.

维护 s[u][i] 表示节点u的第i位到根结点一共有多少个1.

则路径u,v上的第i位1的数目为 (设 t = lca(u, v))
`s[u][i]+s[v][i] - 2 * s[t][i] + ((a[t] & (1 << i)) ? 1 : 0)`

```c++
void solve(int tt) {
    cin >> n >> m;
    vector<int> a(n);
    for (auto &x : a) cin >> x;
    LCA g(n);
    for(int i=1;i<n;++i){
        cin>>x>>y;
        x--,y--;
        g.add_edge(x,y);
    }

    vector<vector<int>> s(n,vector<int>(20));
    function<void(int,int)> dfs = [&](int u,int fa) {
        if(fa==-1){
            for(int i=0;i<20;++i){
                if(a[u]&(1<<i)) s[u][i]=1;
                else s[u][i]=0;
            }
        }else{
            for(int i=0;i<20;++i)
                s[u][i]=s[fa][i]+((a[u]&(1<<i))?1:0);
        }
        for(auto v:g.adj[u]){
            if(v!=fa){
                dfs(v,u);
            }
        }

    };

    dfs(0,-1);
    g.build();
    while(m--){
        cin>>x>>y;
        x--,y--;
        int u=g.get_lca(x,y);
        int p=0;
        for(int i=0;i<20;++i){
            if((s[x][i]+s[y][i]-((a[u]&(1<<i))?1:0))%2==1)p=p|(1<<i);
        }
        cout<<p<<"\n";
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

### 异或后最少逆序对数

给你一个有 𝑛 个非负整数组成的数组 𝑎 ，你需要选择一个非负整数 𝑥，对数组 𝑎 的每一个 𝑎𝑖 与 𝑥 进行异或后形成新的数组 𝑏，要求 𝑏 数组的逆序对个数最小，如果有多个 𝑥 满足条件，输出最小的 𝑥。

+ 1 <= n <= 3e5
+ 0 <= ai <= 1e9

**分析**

按位来确定x 的每一位选什么，每一位之间都是独立的，从高到低枚举每一位，如果当前位取1 会使逆序对数量减少就取1，从高位到低位依次确定即可

```c++
long long mergeSort(int l, int r, vector<int>& nums, vector<int>& tmp) {
    if (l >= r) return 0;
    int m = (l + r) / 2;
    long long res = mergeSort(l, m, nums, tmp) + mergeSort(m + 1, r, nums, tmp);
    int i = l, j = m + 1;
    for (int k = l; k <= r; k++) tmp[k] = nums[k];
    for (int k = l; k <= r; k++) {
        if (i == m + 1) nums[k] = tmp[j++];
        else if (j == r + 1 || tmp[i] <= tmp[j]) nums[k] = tmp[i++];
        else {
            nums[k] = tmp[j++];
            res += m - i + 1; //如果是a[i] >= a[j]，tmp[i] <= tmp[j] 改为tmp[i] < tmp[j]
        }
    }
    return res;
}

long long reversePairs(vector<int>& nums) {
    vector<int> tmp(nums.size());
    return mergeSort(0, nums.size() - 1, nums, tmp);
}

void solve(){
    rd(n);
    vector<int> a(n), b;
    rd(a);
    b = a;
    long long mn=reversePairs(a),t;
    long long res=0;
    for(int k=30;~k;--k){
        res |= 1 << k;
        for (int i = 0; i < n; ++i) 
            a[i] = (b[i] ^ res);
        t = reversePairs(a);
        if (t < mn) {
            mn = t;
        } else res ^= 1 << k;
    }
    cout << mn << " " << res << "\n";
}
```

### 方块消失的操作次数

有𝑛堆方块，第𝑖堆方块由ℎ𝑖个方块堆积而成。具体可以看样例。

接下来拆方块。一个方块称为内部方块当且仅当他的上下左右都是方块或者是地面。否则方块就是边界方块。每一次操作都要把边界方块拿掉。

问多少次操作之后所有方块会消失。

<br />
![](/images/posts/leetcode/daimayuan_1.png)
<br />

+ 1 <= n <+ 1e5
+ 1 <= hi <= 1e9

**分析**

我们可以发现，操作最多执行 n/2次 ，因为每一次执行操作，都会消去两边的方块。
我们可以讨论，一个方块是因为变成了两边被消去，还是自己消去了。

对于中间的方块，如果左右的方块足够的多，那么最多 h[i]次操作后，这个方块就消去了。

如果一个方块被消去了后，下一步，它旁边的两个方块也会被消去。

所以我们可以考虑记录每个方块是被左边的方块消去，还是右边的方块消去，还是自己消去了。

dp1[i] 表示考虑左边，第 i个方块在 dp1[i] 次操作后被消去。

转移为 dp1[i]=min(dp1[i-1]+1,a[i]) 。

右边同理，取个 min 就是第 i 个方块消去的操作数。

最后记录下可以坚持最长时间的那个方块。


```c++
int calc(vector<int> &a) {
    int n = a.size(), ans = 0;
    vector<int> dp1(n + 1), dp2(n + 1);
    for (int i = 0; i < n; ++i) dp1[i + 1] = min(dp1[i] + 1, a[i]);
    for (int i = n - 1; ~i; --i) dp2[i] = min(dp2[i + 1] + 1, a[i]);
    for (int i = 0; i < n; ++i) ans = max(ans, min(dp1[i + 1], dp2[i]));
    return ans;
}
```

### 工作安排

有n项工作，每项工作话费一个单位时间，从时刻0开始，你每个时刻可以选择1-n项工作的任意一项工作完成。
每项工作有一个截止日期di, 完成该工作可以获得利润pi，在给定工作利润和截止时间下，能获得的最大利润是多少。

+ 1 <= n <= 1e5
+ 0 <= di, pi <= 1e9

**反悔贪心**

因为我们的工作的时间都为 1 ，所以我们可以直接贪心。用一个优先队列维护。

只要可以放，就放，如果放不了，就拿出利润最小的那个，比较最小的这个和当前工作的利润。

最后我们优先队列里面的，就是我们所选择的工作

```c++
/*
a[i][0]: 截止时间， a[i][1]：利润   
*/
long long maxProfit(vector<vector<int>> &a) {
    sort(all(a),[&](auto x, auto y){return x[0] < y[0];});
    priority_queue<int, vector<int>, greater<int>> q;
    for (int i = 0; i < (int)a.size(); ++i) {
        if (a[i][0] <= 0) continue;
        if (a[i][0] > q.size()) q.push(a[i][1]);
        else if (a[i][1] > q.top()) {
            q.pop();
            q.push(a[i][1]);
        }
    }
    long long ans = 0;
    while (q.size()) {
        ans += q.top(); q.pop();
    }
    return ans;
}
```

### 树上三角形数

给一个𝑛个节点的树, 三角果定义为一个包含3个节点的集合, 且他们两两之间的最短路长度𝑎, 𝑏, 𝑐能够构成一个三角形。

计算这棵树上有多少个不同的三角果

+ 1 <= n <= 1e5
+ 1 <= w <= 1e5
+ 1 <= u, v <= n

**1:树形DP**

+ 当三个点在树的一条路径上无解，其他情况均有解.
+ 由于不在一条路径上因此必然存在一个中间结点分别与这些点相连，假设这个点和其他三个点相连的边权分别a,b,c，则a+b+b+c>a+c证明与权值路径权值无关了。

以点u 为中间点的方案数，这样的方案数由两部分构成：

+ u 的两个不同的子树各选一个，非u 的子树里选一个这样是不会构成一条路径的
+ u 的三个不同的子树各选一个，这样也不会构成一条路径

**计算累乘和**

1. 给定数组a，计算(如下两个函数均可)

<br />
![](/images/posts/leetcode/daimayuan_2.png)
<br />

```c++
long long cal1(vector<long long> &a) {
    long long  s1 = 0, s2 = 0;
    for (auto v : a) s1 += v, s2 += v * v;
    return (s1 * s1 - s2) / 2;
}

long long cal2(vector<long long> &a) {
    long long s1 = 0, s2 = 0;
    for (auto &x: a) {
        s1 = s1 + s2 * x;
        s2 += x;
    }
    return s1;
}
```

2. 给定数组a，计算(如下两个函数均可)

<br />
![](/images/posts/leetcode/daimayuan_3.png)
<br />

```c++
long long fun1(vector<long long> &a) {
    long long  s1 = 0, s2 = 0, s3 = 0;
    for (auto v : a) s3 += v;
    for (auto v : a) s1 += 1ll * v * v * v, s2 += 3ll * v * v * (s3 - v);

    return (s3 * s3 * s3 - s1 - s2) / 6;
}

long long fun2(vector<long long> &a) {
    long long s1 = 0, s2 = 0, s3 = 0;
    for (auto &x: a) {
        s3 = s3 + s2 * x;
        s2 = s2 + s1 * x;
        s1 = s1 + x;
    }
    return s3;
}
```


```c++
long long cal1(vector<long long> &a) {
    long long  s1 = 0, s2 = 0;
    for (auto v : a) s1 += v, s2 += v * v;
    return (s1 * s1 - s2) / 2;
}
long long cal2(vector<long long>&a) {
    long long  s1 = 0, s2 = 0, s3 = 0;
    for (auto v : a) s3 += v;
    for (auto v : a) s1 += 1ll * v * v * v, s2 += 3ll * v * v * (s3 - v);

    return (s3 * s3 * s3 - s1 - s2) / 6;
}
void solve(){
    cin>>n;
    vector<vector<int>> g(n);
    f0(n-1){
        rd(x,y,k);
        x--,y--;
        g[x].push_back(y);
        g[y].push_back(x);
    }
    long long c=0;
    vector<long long> s(n);
    function<void(int,int)> dfs=[&](int u,int fa){
        s[u]=1;
        vector<long long> a;
        for(auto&v:g[u]){
            if(v!=fa){
                dfs(v,u);
                a.push_back(s[v]);
                s[u]+=s[v];
            }
        }
        int m=a.size();
        if(m>=2) c+=cal1(a) * (n-s[u]);
        if(m>=3) c+=cal2(a);
    };
    dfs(0,-1);
    wt(c,'\n');
}
```

**数学**

如果将当前点u 去除掉我们可以得到好几颗子树，问题转化为从这些树中选三棵树每个每棵树选一个结点,设每个树的节点数为s[i], 也就是我们可以枚举其中一个点，然后保证这个点前面的递增，后面的递减即可，也就是等价于我们求到了u 这个点对于已经出现的点作为左半部分，当前枚举的作为u，未枚举到的作为右半部分，因此就可以O(n)的求了。

```c++
vector<long long > s(n);
long long c = 0;
function<void(int,int)> dfs=[&](int u,int fa){
    s[u]=1;
    for(auto&v:g[u]){
        if(v!=fa){
            dfs(v,u);
            c += (s[u] - 1) * s[v] * (n - s[u] - s[v]);
            s[u] += s[v];
        }
    }
    
};
dfs(0,-1);
```

### 环上分段和的最大公约数

环上有𝑛个正整数。你能将环切成𝑘段，每段包含一个或者多个数字。

对于一个切分方案，优美程度为每段数字和的最大公约数，你想使切分方案的优美程度最大，对于𝑘=1,2,…,𝑛输出答案。

+ 1 <= n <= 2000
+ 1 <= a[i] <= 5e7

**分析**

+ g 为数组总和的因子，可以枚举 sum 的因子。
+ 如果一个因子可以是切成k段的结果，那么它也可以是切成k-1段的结果



```c++
vector<long long> maxKsegmentGCD(vector<int> &a) {
    int n = a.size();
    vector<long long> s(n + 1), ans(n);
    for (int i = 0; i < n; ++i) {
        s[i + 1] = s[i] + a[i];
    }

    auto cal = [&](long long x) {
        map<long long, int> mp;
        int cot = 0;
        for (int i = 1; i <= n; i++) {
            mp[s[i] % x]++;
            cot = max(cot, mp[s[i] % x]);
        }
        ans[cot - 1] = max(ans[cot - 1], x);
    };
    int sq = sqrt(s[n]);
    for (int i = 1; i <= sq; ++i) {
        if (s[n] % i == 0) {
            cal(i);
            cal(s[n] / i);
        }
    }
    for (int i = n - 2; ~i; --i) {
        ans[i] = max(ans[i + 1], ans[i]);
    }
    return ans;
}
```

### 字典序最小

从序列 𝑀 个数中顺序选出 𝑁 个不同的数, 使得这 𝑁 个数的字典序最小。
其中 1≤𝑎𝑖≤𝑁, 数据保证 [1,𝑁] 范围内每个数至少出现一次。

让你找一个子序列，为 N 的排列，使得字典序最小，保证至少存在一个排列

+ 1 < n <= m <= 1e6

**单调栈**

顺序枚举，对于a[i],如果a[i]已经在栈中，不做处理，否则，我们弹出所有大于a[i]的数，来保证字典序最小
但需要注意，弹出的数需要保证后面还有这个数，不然的话就不满足每个数都出现一次了。

```c++
/*
0 <= a[i] <= n - 1
*/
vector<int> minLexicographicalPerm(vector<int> &a, int n) {
    int m = a.size();
    vector<int> p(n), s, st(n);
    for (int i = 0; i < m; ++i) {
        p[a[i]] = i;
    }
    for (int i = 0; i < m; ++i) {
        if (st[a[i]]) continue;
        while (s.size() && s.back() > a[i] && p[s.back()] > i) {
            st[s.back()] = 0;
            s.pop_back();
        }
        st[a[i]] = 1;
        s.push_back(a[i]);
    }
    return s;
}
```

### 好序列

有一个长为𝑛的序列𝐴1,𝐴2,…,𝐴𝑛。定义一个序列{𝐴}是好的， 当且仅当他的每一个子区间[𝑙,𝑟]满足，至少存在一个元素𝑥仅出现了一次。

+ 1 <= n <= 2e5 
+ 1 <= a[i] <= 1e9

**启发式合并**


```c++
bool isGoodSequence(vector<int> &a) {
    int n = a.size();
    vector<int> pre(n + 1, -1), nxt(n + 1, n + 1);
    map<int, int> mp;
    for (int i = 1; i <= n; ++i) {
        pre[i] = mp[a[i - 1]];
        nxt[mp[a[i - 1]]] = i;
        mp[a[i - 1]] = i;
    }
    function<bool(int, int)> split = [&](int l, int r) -> bool {
        if (l >= r) return 1;
        int x = l, y = r;
        while (x <= y) {
            if (pre[x] < l && r < nxt[x]) return split(l, x - 1) && split(x + 1, r);
            if (pre[y] < l && r < nxt[y]) return split(l, y - 1) && split(y + 1, r);
            x++, y--;
        }
        return 0;
    };
    return split(1, n);
}
```

### 区间和

长度为n的数组A, 给出q个提示，第i个提示是A中L到R连续元素的区间和，能否根据q个提示知道数组所有元素的和？

**分析**

对于给定的区间和，我们考虑前缀和。

给定区间 [l,r] 的和，相当于告诉了我们 s[r] - s[l - 1]的值，如果我们知道了其中一个数的值，那么另外的一个值也可以得到。

我们可以通过给定的关系，得到 s[n] 的值。所以我们直接用一个并查集维护即可。


```c++
struct DSU {
  public:
    DSU() : _n(0) {}
    explicit DSU(int n) : _n(n), parent_or_size(n, -1) {}

    int merge(int a, int b) {
        assert(0 <= a && a < _n);
        assert(0 <= b && b < _n);
        int x = get(a), y = get(b);
        if (x == y) return x;
        if (-parent_or_size[x] < -parent_or_size[y]) std::swap(x, y);
        parent_or_size[x] += parent_or_size[y];
        parent_or_size[y] = x;
        return x;
    }

    bool same(int a, int b) {
        assert(0 <= a && a < _n);
        assert(0 <= b && b < _n);
        return get(a) == get(b);
    }

    int get(int a) {
        assert(0 <= a && a < _n);
        if (parent_or_size[a] < 0) return a;
        return parent_or_size[a] = get(parent_or_size[a]);
    }

    int size(int a) {
        assert(0 <= a && a < _n);
        return -parent_or_size[get(a)];
    }

    std::vector<std::vector<int>> groups() {
        std::vector<int> leader_buf(_n), group_size(_n);
        for (int i = 0; i < _n; i++) {
            leader_buf[i] = get(i);
            group_size[leader_buf[i]]++;
        }
        std::vector<std::vector<int>> result(_n);
        for (int i = 0; i < _n; i++) {
            result[i].reserve(group_size[i]);
        }
        for (int i = 0; i < _n; i++) {
            result[leader_buf[i]].push_back(i);
        }
        result.erase(
            std::remove_if(result.begin(), result.end(),
                           [&](const std::vector<int>& v) { return v.empty(); }),
            result.end());
        return result;
    }

  private:
    int _n;
    // root node: -1 * component size
    // otherwise: parent
    std::vector<int> parent_or_size;
};

bool check(int n, vector<array<int, 2>> &Q) {
    DSU dsu(n + 1);
    for (auto& [l, r]: Q) {
        dsu.merge(l - 1, r);
    }
    if (dsu.same(0, n)) return 1;
    return 0;
}
```

## acwing

### 平均值大于k的最长子数组长度

[acwing 周赛57T3](https://www.acwing.com/problem/content/4490/)


长度为n的数组A, 请你找到一个序列 a 的连续子序列 a[l], a[l+1], ..., a[r]，要求:

+ a[l]+a[l+1]+,,,+a[r] > 100 * (r - l + 1)
+ 子数组长度尽可能大

+ 1 <= n <= 1e6
+ 0 <= a[i] <= 5000

求子数组的最大可能长度. 


**分析**

s[r] - s[l - 1] > 100 (r - l + 1)

即： s[r]-s[l-1]-100(r-l+1) > 0

设 s1 为 a[i] - 100 数组的前缀和，
即 s1[r] > s1[l] (l < r) 求 r - l 的最大值。

如果 i < j 同时 s[i] <= s[j] 则，s[j] 不会成为任意大于j的最小的l，所以可以维护一个单调递减的栈，每次对栈进行二分。


**代码**

```c++
int lengthestSubArray(vector<int> &a, int k) {
    int n = a.size(), ans = 0;
    vector<long long> s(n + 1);
    for (int i = 0; i < n; ++i) 
        s[i + 1] = s[i] + (a[i] - k);
    vector<int> sk;
    for (int i = 0; i <= n; ++i) {
        if (sk.empty() || s[sk.back()] > s[i]) {
            sk.push_back(i);
        } else {
            int l = 0, r = sk.size() - 1, res = -1;
            while (l <= r) {
                int mid = (l + r) / 2;
                if (s[sk[mid]] < s[i]) {
                    res = mid;
                    r = mid - 1;
                } else l = mid + 1;
            }
            if (res != - 1)
                ans = max(ans, i - sk[res]);
        }
    }
    return ans;
}
```

**O(n)算法**

对原数组作减k操作，设s为新数组的前缀和，问题转化为：
在s数组中找到一对i,j 使得 s[j] > s[i] 且 (j - i) 最大。

+ 对于一个i，如果i左侧有小于或等于s[i]的元素，那么i不可能成为最优答案中的i。
+ 对于一个j，如果j右侧有大于或等于s[j]的元素，那么j不可能成为最优答案中的j。

设 lmn[i]：表示 s[0],...s[i]中的最小值，rmx[j]: 表示s[j]...s[n]中的最大值。

显然，lmn 和 rmx 都是单调不增序列。

从前向后遍历两个数组，如果 lmn[i] >= rmx[j]，则执行 i++，
如果 lmn[i] < rmx[j], 更新 ans, 同时执行 j++;


```c++
int lengthestSubArray(vector<int> &a, int k) {
    int n = a.size();
    
    vector<long long> s(n + 1);
    for (int i = 0; i < n; ++i) 
        s[i + 1] = s[i] + (a[i] - k);

    auto lmn = s, rmx = s;
    for (int i = 1; i <= n; ++i) 
        lmn[i] = min(lmn[i - 1], s[i]);
    for (int i = n - 1; i >= 0; --i) 
        rmx[i] = max(rmx[i + 1], s[i]);

    int i = 0, j = 0, ans = 0;
    while (i <= n && j <= n) {
        if (lmn[i] < rmx[j]) {
            ans = max(ans, j - i);
            j = j + 1;
        } else i = i + 1;
    }
    
    return ans;
}
```

### 所有子数组平均数之和

[牛客小白月赛51 F](https://ac.nowcoder.com/acm/contest/11228/F)

给定一个数组,求出这段数组中所有子数组的平均数之和。
答案对1e9+7取模，假设答案的最简分数表示为a/b,你需要输出最小的非负整数x，使得`x*b`与a模(1e9+7)同余。

+ 1 <= n <= 1e6
+ 0 <= a[i] <= 1e9

**分析**

```
对于不同长度的子区间，每个元素贡献的次数

         a[1] a[2] a[3] a[4] a[5] a[6] a[7]
l = 1     1    1    1    1    1    1    1
l = 2     1    2    2    2    2    2    1
l = 3     1    2    3    3    3    2    1
l = 4     1    2    3    4    3    2    1
l = 5     1    2    3    3    3    2    1
l = 6     1    2    2    2    2    2    1
l = 7     1    1    1    1    1    1    1

```

对于长度l=1,2,...,n分别考虑
当l=1时，所有长度为1的子数组和为s[n]-s[0],记作sum[1];
当l=2时，a[1]和a[n]贡献一次，a[2]...a[n-1]贡献两次
...
可得递推关系，见代码。

最后对于每个长度进行累加和， a/b = a* pow(b,mod-2)


```c++
int calSubArrayMeanSum(vector<int> &a) {
    int n = a.size(), mod = 1e9 + 7;
    vector<long long> p(n + 1), s(n + 1);
    for (int i = 0; i < n; ++i)
        p[i + 1] = (p[i] + a[i]) % mod;
    for (int l = 1; l <= n; ++l) {
        if (l <= (n + 1) / 2) s[l] = (s[l-1] + p[n + 1 - l] - p[l - 1]) % mod;
        else s[l] = s[n + 1 - l];
    }

    auto qp = [&](long long x, long long y) {
        long long c = 1, t = x;
        for (; y; y >>= 1) {
            if (y & 1) c = c * t % mod;
            t = t * t % mod;
        }
        return c;
    };

    long long ans = 0;
    for (int l = 1; l <= n; ++l) {
        ans = (ans + s[l] * qp(l, mod - 2)) % mod;
    }
    return (ans + mod) % mod;
}
```

### 均值大于等于k的子数组数目

[atcoder arc075E](https://atcoder.jp/contests/arc075/tasks/arc075_c)

给定长度为n的数组k，求有多少对子数组，其平均值大于等于k。

+ 1 <= n <= 2e5
+ 1 <= k <= 1e9
+ 1 <= a[i] <= 1e9

**分析**

首先对所有数减去k，设其前缀和数组为s,问题转化为有多少对l,r使得s[r]-s[l-1]>=0, 用树状数组对s求有多少个顺序对。

```c++
long long countSubArraysK(vector<int> &a, int k) {
    int n = a.size();
    vector<long long> s(n + 1), tr(n + 1);
    for (int i = 0; i < n; ++i) {
        s[i + 1] = s[i] + (a[i] - k);
    }

    auto add = [&](int x) {
        for (; x <= n + 1; x += x & -x) tr[x - 1] += 1;
    };

    auto ask = [&](int x) {
        int res = 0;
        for (; x > 0; x -= x & -x) res += tr[x - 1];
        return res;
    };

    auto v = s;
    sort(v.begin(), v.end());
    v.erase(unique(begin(v), end(v)), end(v));

    long long ans = 0;
    for (int i = 0; i <= n; ++i) {
        int p = lower_bound(v.begin(), v.end(), s[i]) - v.begin() + 1;
        ans += ask(p); //如果是大于k的数目，改为ask(p-1)
        add(p);
    }
    return ans;
}
```

### 数对

[牛客 河南赛D](https://ac.nowcoder.com/acm/contest/37344/D)

给定长度为n的数列和两个整数x,y. 求有多少个数对 l, r, 满足

+ 1 <= l <= r <= n
+ a[l] + a[l+1] + ,,, + a[r]  <= x + y * (r - l + 1)

+ 1 <= n <= 2e5
+ -1e9 <= a[i] <= 1e9
+ -1e12 <= x, y <= 1e12

**解析**

另 b[i] = a[i] - y, 设b的前缀和数组为s，上述公式变为 s[r] - s[l - 1] <= x
可以在值域上维护一个树状数组，维护每个前缀和数值的个数和。遍历右边端点r进行累加。

```c++
long long countPairs(vector<int> &a, long long x, long long y) {
    int n = a.size();
    vector<long long> s(n + 1), v;
    for (int i = 0; i < n; ++i) {
        s[i + 1] = s[i] + a[i] - y;
    }

    for (int i = 0; i <= n; ++i) {
        v.push_back(s[i]);
        v.push_back(s[i] - x);
    }

    sort(v.begin(), v.end());
    v.erase(unique(begin(v), end(v)), end(v));

    vector<long long> tr(v.size() + 1);

    auto add = [&](int x) {
        for (; x <= (int)tr.size(); x += x & -x) tr[x - 1] += 1;
    };

    auto ask = [&](int x) {
        int res = 0;
        for (; x > 0; x -= x & -x) res += tr[x - 1];
        return res;
    };

    auto get = [&](long long x) {
        return lower_bound(v.begin(), v.end(), x) - v.begin() + 1;
    };

    long long ans = 0;
    for (int i = 0; i <= n; ++i) {
        ans += i - ask(get(s[i] - x) - 1);
        add(get(s[i]));
    }
    return ans;
}
```

**解法2**

可以直接用平衡树求 小于 s[r] - x 的l有多少个。 需要一个支持下表访问的multiset。


```c++
struct mulset {
    // mulset 模板
};

long long countPairs(vector<int> &a, long long x, long long y) {
    int n = a.size();
    vector<long long> s(n + 1), v;
    for (int i = 0; i < n; ++i) {
        s[i + 1] = s[i] + a[i];
    }

    mulset<long long, less<long long>> st;
    st.insert(0);

    long long ans = 0;
    for (int i = 1; i <= n; ++i) {
        ans += i - st.order_of_key(s[i] - x);
        st.insert(s[i]);
    }
    return ans;
}
```
