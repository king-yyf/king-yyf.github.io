---
layout: post
title: 支持下标的平衡树
date: 2021-12-12
tags: 算法专题  
---


===

Index
---
<!-- TOC -->

- [简介](#简介)
- [pbds使用](#pbds使用)
- [pbds实现multiset](#pbds实现multiset)
- [序列顺序查询](#序列顺序查询)
- [滑动窗口中位数](#滑动窗口中位数)
   



<!-- /TOC -->


### 简介

c++ 中的set, multiset, map等 都支持 维持容器的有序以及按序遍历，但是却不支持下标访问，对于一些既要有序又要支持下标访问的问题，可以使用c++中的PBDS（Policy-based data structures）或者 python中的SortedList


### pbds使用


```c++
#include <bits/stdc++.h>
using namespace std;
#include<ext/pb_ds/assoc_container.hpp>
#include<ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;

template<class T>
using ordered_set = tree<T,null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;

int main() {
    vector<int> v {1, 3, 5, 2, 4, 8};
    ordered_set<int> s;
    for (auto x : v) {
        s.insert(x); // 插入元素
    }
    // 查找有多少个数小于x
    cout << s.order_of_key(5) << "\n"; // output: 4   [1,2,3,4] 
    cout << s.order_of_key(6) << "\n"; // output: 5   [1,2,3,4, 5] 

    s.erase(5); // 删除元素
    cout << s.order_of_key(6) << "\n"; // output: 4   [1,2,3,4]

    // s : [1, 2, 3, 4, 8]
    // 查询下标为i处的值 0 <= i < s.size() ,不在范围内返回0
    for (int i = 0; i <= (int)s.size(); ++i) {
        cout << "index " << i << " = " << *s.find_by_order(i) << "\n";
    }

    /*
    index 0 = 1
    index 1 = 2
    index 2 = 3
    index 3 = 4
    index 4 = 8
    index 5 = 0
    */
}
```

### pbds实现multiset

**1.使用less_equal**

`template<class T> using ordered_set = tree<T, null_type, less_equal<T>, rb_tree_tag, tree_order_statistics_node_update>;`

将pbds中的 less<T> 修改为 less_equal<T>, 即可支持 multiset.


**2.实现模板**

使用方法

+ 定义multiset `mulset<long long, less<long long>> s;`
+ 插入元素 `s.insert(x)`
+ 删除元素 `s.erase(x)`
+ 查询有多少个比 x 小的元素 `s.order_of_key(x)`
+ 查询下标为i处的值 0 <= i < s.size() ,不在范围内返回0 `s.find_by_order(i)`
+ 上一个元素 `s.prev(x)`
+ 下一个元素 `s.next(x)`


```c++
template<typename T, typename less>
struct mulset_cmp {
    bool operator () (const pair<T, size_t>& x, const pair<T, size_t>& y) const {
        return less()(x.first, y.first) ? true : (less()(y.first, x.first) ? false : less()(x.second, y.second));
    }
};
 
template<typename T, typename less>
struct mulset {
    tree<pair<T, size_t>, null_type, mulset_cmp<T, less>, rb_tree_tag, tree_order_statistics_node_update> t;
    map<T, size_t> mp;

    void insert(T v) {
        t.insert({v, ++mp[v]});
    }

    void erase(T v) {
        t.erase({v, mp[v]--});
    }

    size_t order_of_key(T v) {
        return t.order_of_key({v, 0});
    }

    T find_by_order(size_t r) {
        return t.find_by_order(r)->first;
    }

    T prev(T v) {
        auto it = t.lower_bound({v, 0});
        return (--it)->first;
    }

    T next(T v) {
        return t.lower_bound({v + 1, 0})->first;
    }
};
```

### 序列顺序查询

[leetcode 5937](https://leetcode-cn.com/problems/sequentially-ordinal-rank-tracker/)

题目大意：设计一个支持查询景点排名的系统，该系统支持add,get两种操作，景点有score，name两种属性，score越大，景点越好，score一样时，name字典序越小，景点越好。

- add: 添加一个景点
- get: 查询景点中第i好的景点，第一次调用查询第1好，第二次查询第2好，以此类推。
- get 和 add 总共调用次数不超过 40000 次

1. 使用c++的 pbds库


```c++
#include<ext/pb_ds/assoc_container.hpp>
#include<ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;

template<class T>
using ordered_set = tree<T,null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;

class SORTracker {
public:
    ordered_set<pair<int,string>> s;
    int cnt = 0;
    SORTracker() {}
    
    void add(string name, int score) {
        s.insert({-score, name});
    }
    
    string get() {
        return s.find_by_order(cnt++)->second;
    }
};

```


2. 使用python 中的 sortedcontainers

```python
from sortedcontainers import SortedList

class SORTracker:

    def __init__(self):
        self.data = SortedList([])
        self.cnt = 0

    def add(self, name: str, score: int) -> None:
        self.data.add((-score, name))

    def get(self) -> str:
        self.cnt += 1
        return self.data[self.cnt - 1][1]
```



### 滑动窗口中位数


[leetcode 480](https://leetcode-cn.com/problems/sliding-window-median/)

中位数是有序序列最中间的那个数。如果序列的长度是偶数，则没有最中间的数；此时中位数是最中间的两个数的平均数。

例如：

[2,3,4]，中位数是 3
[2,3]，中位数是 (2 + 3) / 2 = 2.5
给你一个数组 nums，有一个长度为 k 的窗口从最左端滑动到最右端。窗口中有 k 个数，每次窗口向右移动 1 位。你的任务是找出每次窗口移动后得到的新窗口中元素的中位数，并输出由它们组成的数组。


```python
from sortedcontainers import SortedList

class Solution:
    def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:
        q = SortedList([])
        ans = []
        for i in range(len(nums)):
            q.add(nums[i])
            if i < k - 1:
                continue
            if i >= k:
                q.discard(nums[i - k])
            
            mid = q[k // 2]
            if k % 2 == 0:
                mid = (mid + q[k // 2 - 1]) / 2.0
            ans.append(mid)
        return ans
```


```c++
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
class Solution {
public:
    tree<pair<int, int>, null_type, less<pair<int, int>>, rb_tree_tag, tree_order_statistics_node_update> t;
    vector<double> medianSlidingWindow(vector<int>& nums, int k) {
        queue<pair<int, int>> q;
        vector<double> ans;
        for(int i = 0; i < nums.size(); ++i) {
            q.push({nums[i], i});
            t.insert({nums[i], i});
            if(q.size() >= k) {
                long x = (*t.find_by_order(k / 2)).first;
                long y = (*t.find_by_order((k - 1) / 2)).first;
                ans.push_back((x + y) / 2.0);
                t.erase(q.front());
                q.pop();
            }
        }
        return ans;
    }
};

```