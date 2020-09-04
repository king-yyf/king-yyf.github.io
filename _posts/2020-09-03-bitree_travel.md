---
layout: post
title: 二叉树遍历专题
date: 2020-09-03
tags: leetcode    
---


===

Index
---
<!-- TOC -->

- [先序遍历](#先序遍历)
- [中序遍历](#中序遍历)
- [后序遍历](#后序遍历)
- [二叉树的层次遍历II](#二叉树的层次遍历II)
- [二叉树的锯齿形层次遍历](#二叉树的锯齿形层次遍历)
- [N叉树的层序遍历](#N叉树的层序遍历)
- [二叉树的右视图](#二叉树的右视图)

<!-- /TOC -->



### 先序遍历

递归写法：
```c++
vector<int> res;
vector<int> preorderTraversal(TreeNode* root) {
    if (root) {
        res.push_back(root->val);
        preorderTraversal(root->left);
        preorderTraversal(root->right);
    }
}
```

非递归写法：
```c++
    vector<int> preorderTraversal(TreeNode* root) {
        vector<int> res;
        stack<TreeNode*> stk;
        TreeNode* p = root;
        while (p || !stk.empty()) {
            while (p) {
                res.push_back(p->val);
                stk.push(p);
                p = p->left;
            }
            if (!stk.empty()) {
                p = stk.top();
                stk.pop();
                p = p->right;
            }
        }
        return res;
    }
```

### 中序遍历

递归写法：
```c++
vector<int> res;
vector<int> preorderTraversal(TreeNode* root) {
    if (root) {
        preorderTraversal(root->left);
        res.push_back(root->val);
        preorderTraversal(root->right);
    }
}
```

非递归写法：
```c++
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        stack<TreeNode*> stk;
        TreeNode* p = root;
        while (p || !stk.empty()) {
            while (p) {
                stk.push(p);
                p = p->left;
            }
            if (!stk.empty()) {
                p = stk.top();
                res.push_back(p->val);
                stk.pop();
                p = p->right;
            }
        }
        return res;
    }
```



### 后序遍历

递归写法：
```c++
vector<int> res;
vector<int> preorderTraversal(TreeNode* root) {
    if (root) {
        preorderTraversal(root->left);
        preorderTraversal(root->right);
        res.push_back(root->val);
    }
}
```

非递归写法：
>* pre: root, left, right
>* post: left, right, root
>* root->left->right   ->  root->right->left   ->  left->right->root

```c++
    vector<int> postorderTraversal(TreeNode* root) {
        vector<int> res;
        stack<TreeNode*> stk;
        TreeNode* p = root;
        while (p || !stk.empty()) {
            while (p) {
                stk.push(p);
                res.push_back(p->val);
                p = p->right;
            }
            if (!stk.empty()) {
                p = stk.top();
                stk.pop();
                p = p->left;
            }
        }
        return vector<int>(res.rbegin(), res.rend());
    }
```

### 二叉树的层次遍历II

[leetcode 107](https://leetcode-cn.com/problems/binary-tree-level-order-traversal-ii/)

```c++
    vector<vector<int>> levelOrderBottom(TreeNode* root) {
       if (!root) return {};
        vector<vector<int>> res;
        queue<TreeNode*> q;
        q.push(root);
        
        while (!q.empty()) {
            int sz = q.size();
            vector<int> v(sz);
            
            for (int i = 0; i < sz; ++i) {
                TreeNode* tmp = q.front();
                q.pop();
                v[i] = tmp->val;
                if (tmp->left) q.push(tmp->left);
                if (tmp->right) q.push(tmp->right);
            }
            res.push_back(v);
        }
        reverse(res.begin(), res.end());
        return res;
    }
```


### 二叉树的锯齿形层次遍历

[leetcode 103](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/)

```c++
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        if (!root) return {};
        vector<vector<int>> ans;
        queue<TreeNode*> q;
        q.push(root);
        bool rev = false;
        while (!q.empty()) {
            int sz = q.size();
            vector<int> tmp(sz);
            for (int i = 0; i < sz; ++i) {
                auto p = q.front();
                q.pop();
                if (rev) tmp[sz - 1 - i] = p->val;
                else tmp[i] = p->val;
                if (p->left) q.push(p->left);
                if (p->right) q.push(p->right);
            }
           ans.push_back(tmp);
           rev = !rev;
        }
        return ans;
    }
```

### N叉树的层序遍历

[leetcode 429](https://leetcode-cn.com/problems/n-ary-tree-level-order-traversal/)

```c++
    vector<vector<int>> levelOrder(Node* root) {
        vector<vector<int>> res;
        if(!root)   return res;
        queue<Node*> que;
        que.push(root);
        while(!que.empty()){
            int size=que.size();
            vector<int> tmp;
            while(size--){
                Node *p=que.front();
                que.pop();
                tmp.push_back(p->val);
                for(int i=0;i<p->children.size();i++)
                    que.push(p->children[i]);             
            }
            res.push_back(tmp);
        }
        return res;
    }
```

### 二叉树的右视图

[leetcode 199](https://leetcode-cn.com/problems/binary-tree-right-side-view/)



```c++
    void dfs(TreeNode * root, int lv, vector<int>& res) {
        if (!root) return;
        if (lv >= res.size()) res.push_back(root->val);
        dfs (root->right, lv + 1, res);
        dfs (root->left, lv + 1, res);
    }
    vector<int> rightSideView(TreeNode* root) {
        vector<int> res;
        dfs(root, 0, res);
        return res;
    }
```