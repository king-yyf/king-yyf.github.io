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
- [序列顺序查询](#序列顺序查询)
   



<!-- /TOC -->


### 简介

c++ 中的set, multiset, map等 都支持 维持容器的有序以及按序遍历，但是却不支持下标访问，对于一些既要有序又要支持下标访问的问题，可以使用c++中的PBDS（Policy-based data structures）或者 python中的SortedList



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