---
layout: post
title: 最长上升子串问题
date: 2020-08-15
tags: leetcode    
---


### 1.最长上升子序列

给定一个无序的整数数组，找到其中最长上升子序列的长度。

[leetcode 300](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

```c++
int lengthOfLIS(vector<int>& nums) {
    vector<int> res;
    for (int i = 0; i < nums.size(); ++i) {
        auto it = lower_bound(res.begin(), res.end(), nums[i]);
        if (it == res.end()) res.push_back(nums[i]);
        else *it = nums[i];
    }
    return res.size();
}
```

### 2.最长连续递增序列

给定一个未经排序的整数数组，找到最长且 **连续** 的的递增序列，并返回该序列的长度。

[leetcode 674](https://leetcode-cn.com/problems/longest-continuous-increasing-subsequence/)


```c++
int findLengthOfLCIS(vector<int>& nums) {
    int n = nums.size(), ans = 1;
    if (n <= 1) return n;
    vector<int> dp(n + 1, 1);
    for (int i = 2; i <= n; ++i) {
        if (nums[i - 1] > nums[i - 2]) {
            dp[i] = dp[i - 1] + 1;
        } else dp[i] = 1;
    }
    return *max_element(dp.begin(), dp.end());
}
```

### 3.最长连续序列

给定一个未排序的整数数组，找出最长连续序列的长度。要求算法的时间复杂度为 O(n)。

[leetcode 128](https://leetcode-cn.com/problems/longest-consecutive-sequence/)

```
输入: [100, 4, 200, 1, 3, 2]
输出: 4
解释: 最长连续序列是 [1, 2, 3, 4]。它的长度为 4。
```

```c++
int longestConsecutive(vector<int>& num) {
    unordered_set<int> s(num.begin(), num.end()), searched;
    int longest = 0;
    for (int i: num) {
        if (searched.find(i) != searched.end()) continue;
        searched.insert(i);
        int j = i - 1, k = i + 1;
        while (s.find(j) != s.end()) searched.insert(j--);
        while (s.find(k) != s.end()) searched.insert(k++);
        longest = max(longest, k - 1 - j);
    }
    return longest; 
}
```

### 4.最长递增子序列的个数

给定一个未排序的整数数组，找到最长递增子序列的个数。

[leetcode 673](https://leetcode-cn.com/problems/number-of-longest-increasing-subsequence/)

```c++
int findNumberOfLIS(vector<int>& nums) {
    int maxlen = 1, ret = 0;
    vector<int> cnt(nums.size(), 1), dp(nums.size(), 1);
    for (int i = 1; i < nums.size(); ++i) {
        for (int j = 0; j < i; ++j) {
            if (nums[i] > nums[j]) {
                if (dp[j] + 1 > dp[i]) dp[i] = dp[j]+1, cnt[i] = cnt[j];
                else if (dp[i] == dp[j] + 1) cnt[i] += cnt[j];
            }
        }
        maxlen = max(maxlen, dp[i]);
    }
    for (int i=0;i < dp.size();++i) 
        if (dp[i] == maxlen) ret += cnt[i];
    return ret;
}
```


### 5.最大上升子序列和

给定一个数的序列，求上升子序列中的最大和，例如序列 `(100, 1, 2, 3)`,最大上升子序列和为100。
[noi 3532](http://noi.openjudge.cn/ch0206/3532/)

```c++
int maxLISsum(int a[], int n) { // index begin from 1
    int s[n + 1] = {0}, ans = 0;
    for (int i = 1; i <= n; ++i) {
        s[i] = a[i];
        for (int j = 1; j < i; ++j) {
            if (a[i] > a[j] && s[i] < s[j] + a[i]) 
                s[i] = s[j] + a[i]; 
        }
        ans = max(ans, s[i]);
    }
    return ans;
}
```

### 6.最长公共子上升序列

给定两个整数序列， 求它们的最长上升公共子序列。
[noi 2000](http://noi.openjudge.cn/ch0206/solution/9771864/)

`f(i1,i2)`表示`a1`与`a2[0..i2]`,以`a1[i1]`结尾的最长上升公共子序列 
若`a2[i2] == a1[i1]`,    
`f(i1,i2) = max{ f(i,i2-1) } + 1    (0<=i<i1)`  
若`a2[i2] < a1[i1]`, `f(i1,i2)`不变
若`a2[i2] > a1[i1]`, `f(i1,i2)`不变,
`max{ f(i,i2) }` 的值更新  

```c++
struct node{
    int len;
    vector<int> iv;
    node(){len = 0;}
} ans[maxn], cur; 

vector<int> maxLCSLIS(int a1[], int n1, int a2[], int n2) {
    for (int i = 0; i < n2; ++i) {
        cur.len = 0; cur.iv.clear();
        for (int j = 0; j < n1; ++j) {
            if (a2[i] > a1[j] && ans[j].len > cur.len) cur = ans[j];
            if (a2[i] == a1[j]) {
                ans[j] = cur; ans[j].len++;
                ans[j].iv.push_back(a1[j]);
            }
        }
    }
    int idx = 0;
    for (int i = 1; i < n1; ++i) {
        if (ans[idx].len < ans[i].len) idx = i;
    }
    return ans[idx].iv;
}
```