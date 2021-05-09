---
layout: post
title: 二叉搜索树
date: 2020-07-04
tags: 面试算法    
---

二叉搜索树是一种非常重要的数据结构，这里实现其结构的定义以及插入、查找和删除操作。


### 二叉搜索树定义

**定义**
二叉搜索树是指一颗空树或者具有下列性质的二叉树：

>* 若其左子树存在，则其左子树中所有节点的值都不大于该节点值；
>* 若其右子树存在，则其右子树中所有节点的值都不小于该节点值。 

二叉搜索树的 **中序遍历** 结果为升序排列。

其结构体定义
```c++
struct Node{
    int val;
    Node* left;
    Node* right;
    Node(int v, Node* l = NULL, Node* r = NULL):val(v), left(l), right(r){}
};
```

### 二叉搜索树插入节点

**分析：** 向二叉树中插入节点时，如果待插入数值小于根节点，则将其插入左子树，如果待插入节点值大于根节点，则将其插入右子树，对于相等情况，有两种处理方法：
>* 1. 在`Node`结构中维护一个`count` 成员变量，统计该节点相等数值出现的次数，每次插入操作时初始化为1，遇到相等元素时执行`count++`.
>* 2. 插入其左子树或右子树。  

这里假设二叉搜索树中无重复数据，其插入代码如下
```c++
Node* insert(Node* root, int value){
    if(!root) return new Node(value);

    if(value < root->val) root->left = insert(root->left, value);
    if(value > root->val) root->right = insert(root->right, value);
    return root;
}
```

### 二叉搜索树中查询节点

**分析：** 利用二叉搜索树的性质，如果待查询数值小于根节点，则在左子树中晋系行查询，如果待插入节点值大于根节点，则在右子树中进行查询，如果相等则查找到并返回。

实现代码:  
```c++
Node* search(Node* root, int val){
    if(!root || root->val == val) return root;

    if(root->val > val)
        return search(root->left, val);

    return search(root->right, val);
}
```

### 删除二叉树中的节点

**分析：** 删除节点时，如果待删除节点为叶子结点，则可以直接删除；如果节点左子树和右子树有一个不为空，则将不为空的节点直接替换该节点；如果左子树和右子树都不为空，则找到右子树中的最小值（或左子树中的最大值）替换该节点，同时删除右子树中的最小值。

```c++
Node* minNode(Node* node){
    while(node->left)
        node = node->left;
    return node;
}

Node* delete(Node* node, int value){
    if(!node) return node;

    if(value < node->val){
        node->left = delete(node->left, value);
    }else if(value > node->val){
        node->right = delete(node->right, value);
    }else{
        if(!node->left) return node->right;
        if(!node->right) return node->left;

        Node* min_node = minNode(node->right);
        node->val = min_node->val;
        node->right = delete(node->right, min_node->val);
    }

    return node;
}
```


### 二叉搜索树的最近公共祖先（leetcode 235）

**题目描述：** 给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

**分析：** 
>* 如果p q的值都小于当前节点的值，则递归进入当前节点的左子树；
>* 如果p q的值都大于当前节点的值，则递归进入当前节点的右子树；
>* 如果当前节点的值在p q两个节点的值的中间，那么这两个节点的最近公共祖先则为当前的节点。

```c++
struct TreeNode{
    int val;
    TreeNode* left,
    TreeNode* right;
    TreeNode(int x):val(x), left(NULL), right(NULL){}
};

TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q){
    if (!root) return root;

    if (p->val > root->val && q->val > root->val)
        return lowestCommonAncestor(root->right, p, q);
    else if (p->val < root->val && q->val < root->val)
        return lowestCommonAncestor(root->left, p, q);

    return root;
}
```

### 普通二叉树的最近公共祖先（leetcode 236）

**题目描述：** 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

**分析：** 
>* 递归查询两个节点p q，如果某个节点等于节点p或节点q，则返回该节点的值给父节点。
>* 如果当前节点的左右子树分别包括p和q节点，那么这个节点必然是所求的解。
>* 如果当前节点有一个子树的返回值为p或q节点，则返回该值。（告诉父节点有一个节点存在其子树中）
>* 如果当前节点的两个子树返回值都为空，则返回空指针。

```c++
struct TreeNode{
    int val;
    TreeNode* left,
    TreeNode* right;
    TreeNode(int x):val(x), left(NULL), right(NULL){}
};

TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q){
    if(!root || root == p || root == q) return root;

    TreeNode* pLeft = lowestCommonAncestor(root->left, p, q);
    TreeNode* pRight = lowestCommonAncestor(root->right, p, q);
    if(!pRight) return pLeft;
    if(!pLeft) return pRight;
    return root;
}
```

