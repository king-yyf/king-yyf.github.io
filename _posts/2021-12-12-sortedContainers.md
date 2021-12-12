---
layout: post
title: python 刷题库函数
date: 2021-12-12
tags: 算法专题  
---


===

Index
---
<!-- TOC -->

- [sortedcontainers](#sortedcontainers)
    - [SortedSet](#sortedset)
    - [SortedList](#sortedlist)
    - [SortedDict](#sorteddict)
- [collections](#collections)
    - [Counter](#counter)
- [示例题目]
    - [滑动窗口中位数](#滑动窗口中位数)

   



<!-- /TOC -->


## sortedcontainers

python 中的 sortedcontainers 模块 封装了对列表、字典、集合的排序等常用操作，在算法比赛中非常有用，很多在c++中实现较为困难的题在sortedcontainers的帮助下，python可以很容易完成代码编写。 纯python编写，速度与c扩展一样快，

安装

```
pip install sortedcontainers
```


### sortedset

[API doc](http://www.grantjenks.com/docs/sortedcontainers/sortedset.html)


**常用基本方法**

```python
from sortedcontainers import SortedSet
s = SortedSet([1,2,3,4])

print(3 in ss)        # True
print(len(s))         # 4
s = set([1,2,3,4])    # True
 
s.add(5)          # 添加元素5 s = [1,2,3,4,5]  O(log(n))
s.add(3)          # s = [1,2,3,4,5] 

s.discard(3)      # 删除值为3的元素 s = [1,2,4,5]   O(log(n))

print(s[2])       # s[2] = 4   runtime complexity: O(log(n))

del s[2]          # 删除下标2处元素 O(log(n)) s = [1,2,5]


a = s.pop()       # 取最后一个元素，并删除该元素 O(log(n))
a = s.pop(2)      # 取下标为2的元素，并删除该元素

s.remove(3)       # 删除值为3的元素，不存在时会报异常

s.clear()         # 删除所有元素。O(n)

```


常用集合运算

- `-`  集合差
- `-=` 求差并更新
- `&` 集合交集
- `&=` 求交集并更新
- `^` 对称差 (并集-交集)
- `^=` 异或并更新
- `|` 并集
- `|=` 并集并更新  


**difference**

返回所有在该集合中而不在另一个集合中的元素，返回一个新的SortedSet

```python
from sortedcontainers import SortedSet

ss = SortedSet([1, 2, 3, 4, 5])
ss.difference([4, 5, 6, 7])    # SortedSet([1, 2, 3])
```

**difference_update**

返回所有在该集合中而不在另一个集合中的元素,并更新为当前集合

```python
ss = SortedSet([1, 2, 3, 4, 5])
_ = ss.difference_update([4, 5, 6, 7])  # [1,2,3]
print(ss)      # [1,2,3]
```


**intersection**

返回两个或多个集合的交集, 返回一个新的SortedSet

```python
ss = SortedSet([1, 2, 3, 4, 5])
ss.intersection([4,5,6,7])    # SortedSet([4, 5])
```

**intersection_update**

求两个或多个集合的交集, 并更新该集合为交集的集合

```python
ss = SortedSet([1, 2, 3, 4, 5])
a = ss.intersection_update([4, 5, 6, 7]) #[4,5]
print(ss)       # (4,5)
```


**symmetric_difference**

返回两个集合的异或 (并集-交集) 等价于 `ss ^ s`

```python
ss = SortedSet([1, 2, 3, 4, 5])
s = SortedSet([4,5,6,7])
a = ss.symmetric_difference(s)
b = ss ^ s  # [1,2,3,6,7]
print(a)    # [1,2,3,6,7]
```

**symmetric_difference_update**

返回两个集合的异或,同时更新当前集合


```python
ss = SortedSet([1, 2, 3, 4, 5])
_ = ss.symmetric_difference_update([4, 5, 6, 7])
print(ss)   # [1,2,3,6,7]
```


**union**

集合合并 等价于 `ss | s`

**update** 

集合合并 并更新 等价于 `ss |= s`

**count**

返回某个值在集合中出现次数


### sortedlist

[SortedList API](http://www.grantjenks.com/docs/sortedcontainers/sortedlist.html)

常用函数

- add(value) 添加元素 O(log(n))
- update(iterable) 添加list O(k*log(n))
- clear() 清空list O(n)
- discard(value) 删除值为value的元素，如果不存在，do nothing O(log(n))
- remove(value)  删除值为value的元素，不存在时抛出异常 O(log(n))
- pop(index=-1) 删除并返回下标为的元素，默认-1 O(log(n))
- count(value) 某个值的出现次数 O(log(n))
- index(value,start=None, stop=None) 返回在区间内值第一次出现的下标 O(log(n))
- 支持 in, len, [], del, + 等关键字

**bisect_left(value)**

返回一个大于等于该元素的最小的下标，类似于lower_bound


```python
from sortedcontainers import SortedList
sl = SortedList([10, 11, 15, 16, 19])
sl.bisect_left(14)   # 2
sl.bisect_left(11)   # 1

```

**bisect_right(value)**

返回一个大于该元素的最小的下标，类似于upper_bound

```python
from sortedcontainers import SortedList
sl = SortedList([10, 11, 15, 16, 19])
sl.bisect_right(14)    # 2
sl.bisect_right(11)    # 2
```

**irange(minimum=None, maximum=None, inclusive=True, True, reverse=False)**


返回一个值在[min,max]区间的新iterator 


```python
sl = SortedList([10, 11,12, 15, 16, 19])
it = sl.irange(11, 15)
print(list(it))  # [11,12,15]
```

**islice(start=None, stop=None, reverse=False)**

返回下标从[start,stop)的区间的iterator

```python
sl = SortedList('abcdefghij')
it = sl.islice(2, 6)
print(list(it))   # ['c', 'd', 'e', 'f']
```

**SortedKeyList**

SortedList 的子类，根据key函数去比较元素大小

SortedList中可用的所有相同方法也可在SortedKeyList中使用

```python
from operator import neg
neg(1)   # -1

skl = SortedKeyList(key=neg)

skl = SortedKeyList([3, 1, 2], key=neg)  # [3,2,1]
```

### sorteddict

[sorteddict API](#http://www.grantjenks.com/docs/sortedcontainers/sorteddict.html)

Sorted dict 实现了

- SortedDict
- SortedKeysView
- SortedItemView
- SortedValueView

**初始化**

```python
d = {'alpha': 1, 'beta': 2}
s = SortedDict(d)

s = SortedDict([('alpha', 1), ('beta', 2)])

s = SortedDict({'alpha': 1, 'beta': 2})

s = SortedDict(alpha=1, beta=2)

sd = SortedDict()
sd['c'] = 3
```

**setdefault(key, default=None)**

如果key存在，返回value，如果key不存在，插入{key:default}，返回default

 O(log(n))


```python
sd = SortedDict()
sd.setdefault('a', 1)   # 1
sd.setdefault('a', 10)  # 1
```

常用函数

- pop(key, default=<not-given>) 返回key对应value并删除key
- popitem(index=- 1) 返回下标index处的(key, value)对，并删除
- peekitem(index=- 1) 返回下标index处的(key, value)对
- get(key, default=<not-given>) 返回key对应value，不存在返回default
- keys(),items(),values() 同dict 


## collections

python 中的高性能容器数据结构,包含

- namedtuple(): 创建命名元组子类的工厂函数
- deque 类似列表(list)的容器，实现了在两端快速添加(append)和弹出(pop)
- ChainMap 类似字典(dict)的容器类，将多个映射集合到一个视图里面
- Counter 字典的子类，提供了可哈希对象的计数功能
- OrderedDict 字典的子类，保存了他们被添加的顺序
- defaultdict 字典的子类，提供了一个工厂函数，为字典查询提供一个默认值
- UserDict 封装了字典对象，简化了字典子类化
- UserList 封装了列表对象，简化了列表子类化
- UserString 封装了字符串对象，简化了字符串子类化


### counter

一个计数器工具提供快速和方便的计数。比如

```python
cnt = Counter()
for word in ['red', 'blue', 'red', 'green', 'blue', 'blue']:
    cnt[word] += 1
print(cnt)   # Counter({'blue': 3, 'red': 2, 'green': 1})
```

**elements**

返回一个迭代器，其中每个元素将重复出现计数值所指定次

```python
c = Counter(a=4, b=2, c=0, d=-2)
sorted(c.elements())
# ['a', 'a', 'a', 'a', 'b', 'b']
```

**most_common**

返回一个列表，其中包含 n 个最常见的元素及出现次数，按常见程度由高到低排序。 如果 n 被省略或为 None，most_common() 将返回计数器中的 所有 元素。 计数值相等的元素按首次出现的顺序排序：

```python
Counter('abracadabra').most_common(3)
# [('a', 5), ('b', 2), ('r', 2)]
```

**total()**

计算总计数值。

```python
c = Counter(a=10, b=5, c=0)
c.total()   # 15
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