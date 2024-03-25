---
layout: post
title: 排序数组系列
date: 2020-08-17
tags: 面试算法    
---

===

Index
---
<!-- TOC -->

- [搜索旋转排序数组](#搜索旋转排序数组)
- [搜索旋转排序数组II（有重复）](#搜索旋转排序数组2)
- [寻找旋转排序数组中的最小值](#寻找旋转排序数组中的最小值)
- [寻找旋转排序数组中的最小值（有重复）](#寻找旋转排序数组中的最小值2)
- [寻找两个正序数组的中位数](#寻找两个正序数组的中位数)
- [两个排序数组第k大数](#两个排序数组第k大数)
- [合并k个有序数组](#合并k个有序数组)


<!-- /TOC -->


### 搜索旋转排序数组

[leetcode 33](https://leetcode.cn/problems/search-in-rotated-sorted-array/description/)


假设按照升序排序的数组在预先未知的某个点上进行了旋转。

( 例如，数组 `[0,1,2,4,5,6,7]` 可能变为 `[4,5,6,7,0,1,2]` )。

搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。

**你可以假设数组中不存在重复的元素。**

你的算法时间复杂度必须是 `O(log n)` 级别。

**分析**

chk(mid): a[mid] 是否等于target，或者在targe的右侧，分情况二分即可。

```c++
    int search(vector<int>& a, int t) {
        int n = a.size();
        int l = 0, r = n - 1, ans = n - 1;

        auto chk = [&](int mid) {
            if (a[mid] < a[n - 1]) {
                return t <= a[mid] || t > a[n - 1];
            } else {
                return t <= a[mid] && t > a[n - 1];
            }
        };

        while (l <= r) {
            int mid = (l + r) / 2;
            if (chk(mid)) {
                ans = mid;
                r = mid - 1;
            } else l = mid + 1;
        }

        return a[ans] == t ? ans : -1;
    }
```

### 搜索旋转排序数组2

[leetcode 81](https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/)

假设按照升序排序的数组在预先未知的某个点上进行了旋转。

( 例如，数组 `[0,1,2,4,5,6,7]` 可能变为 `[4,5,6,7,0,1,2]` )。

搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。

最坏时间复杂度为 o(n)

**数组中可能存在重复的元素。**

```c++
    bool search(vector<int>& nums, int target) {
        int l = 0, r = nums.size() - 1;
        while (l <= r) {
            int mid = l + ((r - l) >> 1);
            if (nums[mid] == target) return true;
            else if (nums[mid] > nums[l]) {
                if (nums[l] <= target && target <= nums[mid]) {
                    r = mid - 1;
                } else {
                    l = mid + 1;
                }
            } else if (nums[l] == nums[mid]) {
                l++;
            } else {
                if (nums[mid] <= target && target <= nums[r]) {
                    l = mid + 1;
                } else {
                    r = mid - 1;
                }
            }
        }
        return false;
    }
```

### 寻找旋转排序数组中的最小值

[leetcode 153](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/description/)

假设按照升序排序的数组在预先未知的某个点上进行了旋转。

( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。

请找出其中最小的元素。

**你可以假设数组中不存在重复元素。**

**分析**

显然，前半段都大于最后一个元素，后半段都小于等于最后一个元素，本质是找数组中小于等于最后一个元素的第一个位置。

```c++
    int findMin(vector<int>& a) {
        int n = a.size();
        int l = 0, r = n - 1, ans = a[n - 1];

        while (l <= r) {
            int mid = (l + r) / 2;
            if (a[mid] < a[n - 1]) {
                ans = a[mid];
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        return ans;
    }
```


### 寻找旋转排序数组中的最小值2

[leetcode 154](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/)

假设按照升序排序的数组在预先未知的某个点上进行了旋转。

( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。

请找出其中最小的元素。

**数组中可能存在重复元素。**


```c++
    int findMin(vector<int>& a) {
        int l = 0, r = a.size() - 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (a[mid] < a[r]) r = mid;
            else if (a[mid] > a[r]) l = mid + 1;
            else r--;
        }
        return a[l];
    }
```


### 寻找两个正序数组的中位数

[leetcode 4](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)

给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。

```c++
double findMedianSortedArrays(vector<int>& a1, vector<int>& a2) {
    int n1 = a1.size(), n2 = a2.size();
    if(n1 > n2) return findMedianSortedArrays(a2, a1);
    
    int lo = 0, hi = n1; // range of a1 cut location: n1 means no right half for a1
    while (lo <= hi) {
        int cut1 = (lo + hi)/2; // cut location is counted to right half
        int cut2 = (n1 + n2)/2 - cut1;
        
        int l1 = cut1 == 0? INT_MIN : a1[cut1-1];
        int l2 = cut2 == 0? INT_MIN : a2[cut2-1];
        int r1 = cut1 == n1? INT_MAX : a1[cut1];
        int r2 = cut2 == n2? INT_MAX : a2[cut2];
        
        if (l1 > r2) hi = cut1-1;
        else if (l2 > r1) lo = cut1+1;
        else return (n1+n2)%2? min(r1,r2) : (max(l1,l2) + min(r1,r2))/2.;
    }
    return -1;
}
```

### 两个排序数组第k大数

两个数组从小到大排序，查找两个数组的第k小数

```c++
    int findKthElm(vector<int>& nums1, vector<int>& nums2, int k){
        int le = max(0, int(k-nums2.size())), ri = min(k, int(nums1.size()));
        while(le < ri){
            int m = le + (ri - le)/2;
            if(nums2[k-m-1] > nums1[m])
                le = m + 1;
            else
                ri = m;
        }
        int nums1LeftMax = le == 0 ? INT_MIN:nums1[le-1];
        int nums2LeftMax = le == k ? INT_MIN:nums2[k-le-1];
        return max(nums1LeftMax, nums2LeftMax);
    }
```

### 合并k个有序数组

[lintcode 486](https://www.lintcode.com/problem/merge-k-sorted-arrays/)

将 k 个有序数组合并为一个大的有序数组。


```c++
    struct Node {
        int row, col, val;
        Node(int r, int c, int v): row(r), col(c), val(v) {};
        bool operator < (const Node &obj) const {
            return val > obj.val;
        }
    }; 
    vector<int> mergekSortedArrays(vector<vector<int>> &arr) {
        vector<int> res;
        if (arr.empty()) return res;
        priority_queue<Node> q;
        for (int i = 0; i < arr.size(); ++i) {
            auto v = arr[i];
            if (!v.empty())
                q.push({i, 0, v[0]});
        }

        while (!q.empty()) {
            Node cur = q.top();
            q.pop();
            res.push_back(cur.val);
            if (cur.col + 1 < arr[cur.row].size()) 
                q.push({cur.row, cur.col + 1, arr[cur.row][cur.col + 1]});
        }
        return res;
    }
```