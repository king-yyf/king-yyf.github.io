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

**方法一**：中心扩展方法
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

**方法二**：动态规划法  
`dp[i][j]` 表示从i到j是否为回文子串。

```c++
string longestPalindrome(string s) {
    int n = s.size();
    vector<vector<int>> dp(n, vector<int>(n));
    string ans;
    for (int l = 0; l < n; ++l) {
        for (int i = 0; i + l < n; ++i) {
            int j = i + l;
            if (l == 0) dp[i][j] = 1;
            else if (l == 1)  dp[i][j] = (s[i] == s[j]);
            else   dp[i][j] = (s[i] == s[j] && dp[i + 1][j - 1]);
            if (dp[i][j] && l + 1 > ans.size())
                ans = s.substr(i, l + 1);
        }
    }
    return ans;
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

### 5.分割回文串

[leetcode 131](https://leetcode-cn.com/problems/palindrome-partitioning/)

给定一个字符串 s，将 s 分割成一些子串，使每个子串都是回文串。

返回 s 所有可能的分割方案。

```c++
bool isPalindrome(const string& s, int l, int r) {
    while (l <= r) 
        if (s[l++] != s[r--]) return false;
    return true;
}
void dfs(int idx, string& s,vector<string>& path, vector<vector<string> >& ret) {
    if (idx == s.size()) {
        ret.push_back(path);
        return;
    }
    for (int i = idx; i < s.size(); ++i) {
        if (isPalindrome(s, idx, i)) {
            path.push_back(s.substr(idx,i - idx + 1));
            dfs(i + 1, s, path, ret);
            path.pop_back();
        }
    }
}
vector<vector<string>> partition(string s) {
    vector<vector<string>> ret;
    if (s.empty()) return ret;
    vector<string> path;
    dfs(0, s, path, ret);
    return ret;
}
```

### 6.让字符串成为回文串的最少插入次数

[leetcode 1312](https://leetcode-cn.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/)  

给你一个字符串 s ，每一次操作你都可以在字符串的任意位置插入任意字符。

请你返回让 s 成为回文串的 **最少操作次数** 。

**分析**：`dp[i][j]` 表示对于字符串 s 的子串 `s[i:j]`（这里的下标从 0 开始，并且 s[i:j] 包含 s 中的第 i 和第 j 个字符），最少添加的字符数量，使得 s[i:j] 变为回文串。  

则：`dp[i][i] = 0`, 状态转移方程为：

```c++
if (s[i] != s[j])
    dp[i][j] = min(dp[i + 1][j] + 1, dp[i][j - 1] + 1)         
if (s[i] == s[j])
    dp[i][j] = min(dp[i + 1][j] + 1, dp[i][j - 1] + 1, dp[i + 1][j - 1])   
```
代码：  
```c++
int minInsertions(string s) {
    int n = s.size();
    vector<vector<int>> dp(n, vector<int>(n));
    for (int len = 2; len <= n; ++len) {
        for (int i = 0; i <= n - len; ++i) {
            int j = i + len - 1;
            dp[i][j] = min(dp[i + 1][j], dp[i][j - 1]) + 1;
            if (s[i] == s[j]) {
                dp[i][j] = min(dp[i][j], dp[i + 1][j - 1]);
            }
        }
    }
    return dp[0][n - 1];
}
```
