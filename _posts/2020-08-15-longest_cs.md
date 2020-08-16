---
layout: post
title: 公共子串，回文子串问题
date: 2020-08-15
tags: leetcode    
---


### 1.最长公共子序列

给定两个字符串 s 和 t，返回这两个字符串的最长公共子序列的长度。

[leetcode 1143](https://leetcode-cn.com/problems/longest-common-subsequence/)

```c++
int longestCommonSubsequence(string s, string t) {
    int n = s.size(), m = t.size();
    vector<vector<int>> dp(n + 1, vector<int>(m + 1, 0));
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            if (s[i - 1] == t[j - 1])  
                dp[i][j] = dp[i - 1][j - 1] + 1;
            else dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
        }
    }
    return dp[n][m];
}
```

### 2.最长回文子序列

[leetcode 516](https://leetcode-cn.com/problems/longest-palindromic-subsequence/)

给定一个字符串 s ，找到其中最长的回文子序列，并返回该序列的长度。可以假设 s 的最大长度为 1000 。

```c++
int longestPalindromeSubseq(string s) {
    int n = s.size();
    vector<vector<int>> f(n, vector<int>(n));
    for (int i = n - 1; i >= 0; --i) {
        f[i][i] = 1;
        for (int j = i + 1; j < n; ++j) {
            if (s[i] == s[j]) 
                f[i][j] = f[i + 1][j - 1] + 2;
            else 
                f[i][j] = max(f[i + 1][j], f[i][j - 1]);
        }
    }
    return f[0][n - 1];
}
```

### 3.最长回文子串

给定一个字符串 s，找到 s 中最长的回文子串。

[leetcode 5](https://leetcode-cn.com/problems/longest-palindromic-substring/)

```c++
string longestPalindrome(string s) {
    if (s.size() < 2) return s;
    int n = s.size(), max_l = 0, max_len = 1, l, r;
    for (int i = 0; i < n && n - i > max_len / 2;) {
        l = r = i;
        while (r < n - 1 && s[r + 1] == s[r]) ++r;
        i = r + 1;
        while (r < n - 1 && l > 0 && s[r + 1] == s[l - 1]) {
            ++r; --l;
        }
        if (max_len < r - l + 1) {
            max_len = r - l + 1;
            max_l = l;
        }
    }
    return s.substr(max_l, max_len);
}
```

### 4.回文子串数目

[leetcode 647](https://leetcode-cn.com/problems/palindromic-substrings/)

给定一个字符串，你的任务是计算这个字符串中有多少个回文子串。

具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被计为是不同的子串。

```c++
int countSubstrings(string s) {
        int ans = s.size();
        for (float center = 0.5; center < s.size(); center += 0.5) {
            int left = int(center - 0.5), right = int(center + 1);
            while(left >= 0 && right < s.size() && s[left--] == s[right++]) 
                ans++;
        }
        return ans;
    }
```

