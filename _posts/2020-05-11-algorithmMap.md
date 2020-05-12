---
layout: post
title: 海量数据面试算法总结 
date: 2020-05-10 
tags: 算法    
---

# 本文案由 夕小瑶的卖萌屋 整理
算法工程师思维导图

## 数学基础

### 线性代数

https://www.zhihu.com/question/21082351/answer/1023650088

- 行列式

  https://www.zhihu.com/question/36966326/answer/69790713

- 矩阵运算

  https://zhuanlan.zhihu.com/p/67943590

### 概率论与统计

- 排列组合

  https://zhuanlan.zhihu.com/p/76305844
  好：https://zhuanlan.zhihu.com/p/105709399

- 概率计算

    - 一根绳子被切两刀能组成一个三角形的概率

      https://blog.csdn.net/hefenglian/article/details/82463746

    - 一段绳子切n刀，组成n边型

      https://www.zhihu.com/question/25408010

    - 抛硬币多少次才连续两次正面朝上

      https://zhuanlan.zhihu.com/p/68358814

    - 圆上取3点组成锐角三角形

      https://zhuanlan.zhihu.com/p/69530841

    - 分配白球和红球，取到红球概率

      https://blog.csdn.net/sysysty/article/details/52891663

    - 递归求期望

      https://blog.csdn.net/pure_life/article/details/8100984

    - 地区每天下雨的概率是0.8，天气预报准确性为0.8

      https://www.zhihu.com/question/41438692

- 统计

    - 期望与方差

      常见分布：
      https://blog.csdn.net/Ga4ra/article/details/78935537
      
      期望与方差的推导：
      https://zhuanlan.zhihu.com/p/30496723

### 智力题

https://zhuanlan.zhihu.com/p/106071158

- 取物游戏必胜

  https://blog.csdn.net/lilinfang90/article/details/23481511

## 数据结构与算法

### 数据结构

- 栈

    - 二维矩阵/直方图最大距形面积

      矩阵最大面积：
      https://leetcode-cn.com/problems/maximal-rectangle/solution/zui-da-ju-xing-by-leetcode/
      直方图dp

    - 最小栈

      https://leetcode-cn.com/problems/min-stack/solution/xiang-xi-tong-su-de-si-lu-fen-xi-duo-jie-fa-by-38/

- 链表

    - LRU：双向链表

      https://leetcode-cn.com/problems/lru-cache/
      
      Least Recently Used

    - 寻找重复数-环的入口

      https://leetcode-cn.com/problems/find-the-duplicate-number/
      
      快慢指针，相遇后，从头发起一个指针，按相同速度走，相遇即使环的入口

- 并查集

  http://www.cnblogs.com/cyjb/p/UnionFindSets.html

    - 字节跳动大闯关

      https://blog.csdn.net/sinat_27705993/article/details/82053102
      
      求不同并查集的个数
      多一个count，每union一次count就减1

    - 岛屿个数

      https://leetcode-cn.com/problems/number-of-islands/solution/dfs-bfs-bing-cha-ji-python-dai-ma-java-dai-ma-by-l/
      
      多开一个放0的岛

- 树

    - 二叉树前序/中序/后序

      前序： = 自顶向下
      curr=stack.pop()
      print(curr.val)
      stack.push(curr.right)
      stack.push(curr.left)
      
      中序：加一个记录
      curr = stack.pop()
      if curr in cache:
        res.append(curr.val)
        continue
      cache.add(curr)
      stack.append(curr.right)
      stack.append(curr)
      stack.append(curr.left)
      
      后序：同中序 = 自底向上

    - 二叉树的最近公共祖先

      三个条件满足两个就是True：
      1.左子树包含p1或p2
      2.右子树包含p1或p2
      3.自己是p1或p2

    - 二分查找树

      AVL：https://mp.weixin.qq.com/s/dYP5-fM22BgM3viWg4V44A
      
      为啥有了BST和AVL还需要红黑树？https://zhuanlan.zhihu.com/p/72505589
      AVL每次进行插入/删除节点的时候，几乎都会破坏平衡树的第二个规则，进而我们都需要通过左旋和右旋来进行调整，使之再次成为一颗符合要求的平衡树
      如果在那种插入、删除很频繁的场景中，平衡树需要频繁着进行调整，这会使平衡树的性能大打折扣，为了解决这个问题，于是有了红黑树

    - 二叉树中的最大路径和

      https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/

    - 完全二叉树插入

      https://blog.csdn.net/psc0606/article/details/48742239
      利用完全二叉树的性质，找到要插入的位置，先判断左子树的最右结点与右子树的最右结点高度，如果相等，只需要插入到左子树即可，否则插入右子树

    - 完全二叉树节点数

      https://leetcode-cn.com/problems/count-complete-tree-nodes/solution/er-fen-cha-zhao-by-xu-yuan-shu/

- 哈希表

    - 求最长非重复字符串长度

      https://blog.csdn.net/zd_nupt/article/details/82669299
      
      做个hash_table表，记录每个字符的位置。碰到重复的就求两个重复字符之间的距离

    - 前缀和+哈希表

        - 连续子数组和为k的倍数

          https://leetcode-cn.com/problems/continuous-subarray-sum/solution/lian-xu-de-zi-shu-zu-he-by-leetcode/

        - 和为K的子数组

          https://leetcode-cn.com/problems/subarray-sum-equals-k/
          
          连续区间和为K，用字典存储累计和

### 查找

- 二分查找

    - bug-free写法：左闭右开，先写排除中位数的逻辑

      https://www.zhihu.com/question/36132386/answer/97729337
      lower/upper bound

    - 旋转数组

      - 查找最小值/翻转点
      只判断mid-right是否被翻转
      
      - 查找固定target
      1.判断mid-right是否被翻转，找到升序的方向
      2.跟升序区间的left/right和mid比看在不在，不在就搜索另一个空间
      
      
      - 存在重复，寻找最小值
      当num[mid]==num[right]时，right-=1
      因为左闭右开，mid和right中存在别的值
      [0,1,1,1]
      [1,1,0,1]
      [1, 0, 1, 1, 1]
      [1, 1, 1, 1]
      特殊情况下时间复杂度为O(N)

    - 寻找峰值

      左闭右开，往高的地方走
      https://leetcode-cn.com/explore/learn/card/binary-search/210/template-ii/841/

    - 寻找重复数

      https://leetcode-cn.com/problems/find-the-duplicate-number/solution/xun-zhao-zhong-fu-shu-by-leetcode/

    - 双数组中位数

      https://leetcode-cn.com/problems/median-of-two-sorted-arrays/solution/xiang-xi-tong-su-de-si-lu-fen-xi-duo-jie-fa-by-w-2/

    - 找出第K小的距离对

      https://leetcode-cn.com/problems/find-k-th-smallest-pair-distance/solution/hei-ming-dan-zhong-de-sui-ji-shu-by-leetcode/
      
      二分查找 + 双指针

    - 阶乘函数后K个零

      https://leetcode-cn.com/problems/preimage-size-of-factorial-zeroes-function/solution/jie-cheng-han-shu-hou-kge-ling-by-leetcode/

    - 乘法表中第k小的数

      https://leetcode-cn.com/problems/kth-smallest-number-in-multiplication-table/
      
      给定高度m 、宽度n 的一张 m * n的乘法表，以及正整数k，你需要返回表中第k 小的数字。

- BFS

    - 迷宫中的最短路径

      https://blog.csdn.net/qq_28468707/article/details/102786710

    - 字符串A和B的最小相似度

      https://leetcode-cn.com/problems/k-similar-strings/solution/xiang-si-du-wei-k-de-zi-fu-chuan-by-leetcode/

    - 抖音红人

      DFS：https://blog.csdn.net/anlian523/article/details/82557468
      
      BFS：https://blog.csdn.net/u014253011/article/details/82556976
      对于每个用户，遍历粉丝数（记录visited）

- DFS

    - 八皇后

      https://blog.csdn.net/handsomekang/article/details/41308993
      
      一行一行依次遍历(从上往下),决定放在哪列(从左往右),这样就不用判断行冲突,只需要判断列冲突和主斜线副斜线冲突.
      
      对角线=>斜率为1 => abs(A[i]-A[j])==abs(i-j)

    - 全排列

      https://leetcode-cn.com/problems/permutations/solution/
      
      有重复数字的全排列 sort：https://leetcode-cn.com/problems/permutations-ii/

    - 复原IP地址

      https://blog.csdn.net/OneDeveloper/article/details/84946233
      
      dfs，如果加完3个“.”了则判断是否符合条件，否则继续加（start，start+3）

    - 连通岛屿个数

      字节-部门合并：https://blog.csdn.net/zd_nupt/article/details/82669299
      
      dfs：每次遍历到1，则把联通的岛置为0

- 双指针

    - 两数之和

      https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted/solution/liang-shu-zhi-he-ii-shu-ru-you-xu-shu-zu-by-leetco/
      
      在已排序的数组中找到两个数，和为target
      
      双指针暴力求解 n^2
      字典求解 时间n，空间n
      
      我们使用两个指针，初始分别位于第一个元素和最后一个元素位置，比较这两个元素之和与目标值的大小。如果和等于目标值，我们发现了这个唯一解。如果比目标值小，我们将较小元素指针增加一。如果比目标值大，我们将较大指针减小一。移动指针后重复上述比较知道找到答案。
      时间 n，空间1

    - 数组中两数相减的最大值

      https://blog.csdn.net/fkyyly/article/details/83930343
      
      非排序数组中两个数相减（前面减后面）的最大值。
      i<j, max(a[i]-a[j])
      
      if a[i]-a[j]>0: j++
      else: i = j, j++

- 滑动窗口

    - 最小覆盖子串

      https://leetcode-cn.com/problems/minimum-window-substring/solution/zui-xiao-fu-gai-zi-chuan-by-leetcode-2/

    - 和为K的子数组

      https://blog.csdn.net/a546167160/article/details/94401251
      
      当区间和等于target，再向后遍历，可以i+或j+，但是j+可能会越界，因此选择i+

    - 乘积小于K的子数组

      https://leetcode-cn.com/problems/subarray-product-less-than-k/solution/cheng-ji-xiao-yu-kde-zi-shu-zu-by-leetcode/

### 排序

- 插入

    - 插入排序：稳定

      把后面的某个一次次插到前面，再管后面的，第一次确定的位置可能不是最终位置

    - 希尔排序

- 选择

    - 选择排序

      每次选择最小的放到前面

    - 堆排

- 交换

    - 冒泡排序：稳定

      把某个确定好，再管其他的，第一次确定的位置是最终位置

    - 快速排序

      https://blog.csdn.net/qq_36528114/article/details/78667034
      
      快排优化：
      1. 在个数小于N时使用插入排序
      2. 尾递归优化，减少递归栈的深度
      3. 加入三取样切分 //省去了对重复元素的比较，性能接近线性

        - 快速选择

- 归并排序：稳定

  原地归并：直接把合适的片段swap过去
  https://blog.csdn.net/xiaolewennofollow/article/details/50896881
  两个片段的交换需要三次逆转：分别逆转[1, i]和[i+1,n]
  再逆转[1, n]
  
  大数据归并应用较多

    - 数组中的逆序对

      https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/

    - 区间和的个数在某个区间

      https://leetcode-cn.com/problems/count-of-range-sum/

- 基数排序：稳定
- 链表排序

  https://www.cnblogs.com/TenosDoIt/p/3666585.html
  https://leetcode-cn.com/problems/sort-list/solution/sort-list-gui-bing-pai-xu-lian-biao-by-jyd/

- 拓扑排序

  https://www.cnblogs.com/fengziwei/p/7875355.html

- 字典序

    - 下一个排列

      https://leetcode-cn.com/articles/next-permutation/

    - 字典序的第K小数字

      https://leetcode-cn.com/problems/k-th-smallest-in-lexicographical-order/

    - 字典序排数-先序遍历

      https://leetcode-cn.com/problems/lexicographical-numbers/

    - 按字典序排在最后的子串

      https://leetcode-cn.com/problems/last-substring-in-lexicographical-order/

- TopK问题

    - 移除K位数字得到最小结果-栈

      给定一个以字符串表示的非负整数 num，移除这个数中的 k 位数字，使得剩下的数字最小。
      https://leetcode-cn.com/problems/remove-k-digits/

    - 数组中前K个高频元素

      https://leetcode-cn.com/problems/top-k-frequent-elements/solution/leetcode-di-347-hao-wen-ti-qian-k-ge-gao-pin-yuan-/
      
      哈希统计频率+topK排序
      
      堆排/快速选择/桶排

    - 查找和最小的K对数字

      https://leetcode-cn.com/problems/find-k-pairs-with-smallest-sums/

### 动态规划

- 编辑距离

  https://zhuanlan.zhihu.com/p/80682302

- 最长回文子序列／子串

  最长回文子序列：bbbab -> bbbb
  https://leetcode-cn.com/problems/longest-palindromic-subsequence/solution/dong-tai-gui-hua-si-yao-su-by-a380922457-3/
  
  最长回文子串：bbbab -> bbb
  https://leetcode-cn.com/problems/longest-palindromic-substring/solution/zui-chang-hui-wen-zi-chuan-by-leetcode/

- LCS

  公共子串：要求元素相邻：矩阵最长对角线

- LIS

  1. 排序后求LCS，时间O(n^2)，空间O(n)
  2. dp[i]存储A[:i]的LIS，每个i和前面的下标对比，时间O(n^2)，空间O(n)
  for j in [i-1, 0]:
    if A[j]<A[i]:
      dp[i] = max(dp[i], dp[j]+1)
  3. dp[i]存储LIS为i+1时最大的值，最后len(dp)即为答案，时间O(nlogn)，空间O(n)
  二分法查找插入dp的位置

- 最大子序和

  最大和包含当前和不包含：sum[i] = max(sum[i-1]+a[i], a[i])

- 背包问题

  https://blog.csdn.net/stack_queue/article/details/53544109

- 最短路径

  Dijkstra：单源&边权非负 https://www.jianshu.com/p/ff6db00ad866
  
  Floyd：全源&负环，任意两点间的最短路径，时间复杂度为O(N3)，空间复杂度为O(N2) 
  
  Bellmanford：单源 https://blog.csdn.net/lpjishu/article/details/52413812
  Johnson：全源&非负环

- 股票问题

  https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/solution/yi-ge-fang-fa-tuan-mie-6-dao-gu-piao-wen-ti-by-l-3/

### 模式匹配

- 单模式单匹配：KMP

  字符串匹配，返回第一个匹配的位置
  
  https://www.zhihu.com/question/21923021/answer/281346746

- 多模式单匹配：Trie
- 多模式多匹配：AC自动机

### 大数据

- 蓄水池抽样法

  选k个，新旧元素被选中第概率都是k/n
  第k+1个以k/(k+1)被选中，之前在水池里的被替换概率为k/(k+1)*1/k=1/(k+1)
  则旧元素留下的概率为k/(k+1)，与新元素相等

## 深度学习

### 编码器

- DNN

    - 反向传播
    - 梯度消失与爆炸

      反向传播到梯度消失爆炸
      https://zhuanlan.zhihu.com/p/76772734

        - 原因

          本质上是因为梯度反向传播中的连乘效应
          其实梯度爆炸和梯度消失问题都是因为网络太深，网络权值更新不稳定造成的
          
          激活函数导数*权值<1，多个小于1的数连乘之后，那将会越来越小，导致靠近输入层的层的权重的偏导几乎为0，也就是说几乎不更新，这就是梯度消失的根本原因。
          ，连乘下来就会导致梯度过大，导致梯度更新幅度特别大，可能会溢出，导致模型无法收敛。sigmoid的函数是不可能大于1了，上图看的很清楚，那只能是w了，这也就是经常看到别人博客里的一句话，初始权重过大，一直不理解为啥。。现在明白了。

        - 解决方案

          梯度爆炸：正则化/截断
          梯度消失：
          1.改变激活函数：relu（tanh导数也小于1），但会出现dead relu
          2.batchnorm：使权值w落在激活函数敏感的区域，梯度变化大，避免梯度消失，同时加快收敛
          3.残差结构：求导时总有1在

- CNN

    - 归纳偏置：locality & spatial invariance
    - 1*1卷积核

      作用：1.升维降维(in_channel -> out_channel) 2.非线性
      与全连接层的区别：输入尺寸是否可变，全连接层的输入尺寸是固定的，卷积层的输入尺寸是任意的

    - 反向传播

      通过平铺的方式转换成全联接层
      
      https://zhuanlan.zhihu.com/p/81675803
      
      avg pooling：相当于成了w = [1/4, 1/4, 1/4, 1/4]

    - 稀疏交互与权重共享

      每个输 出神经元仅与前一层特定局部区域内的冲经元存在连接权重
      在卷积神经网络中，卷积核中的 每一个元素将作用于每一次局部输入的特定位置上 
      参数共享的物理意义是使得卷积层具高平移等变性。假如图像中有一 只猫，那么无论百出现在图像中的任何位置 3 我们都应该将 '8i只别为猫
      在猫的 圄片上先进行卷积，再向右平移 l像素的输出，与先将圄片向右平移 J像 素再进行卷积操作的输出结果是相等的。

    - 池化本质：降采样

      平均池化：避免估计方差增大，对背景对保留效果好
      最大池化：避免估计均值偏移，提取纹理信息
      
      油化操作除了能显著降低参数量外，还能够保持对平移、伸缩、旋 转操作的不变性。

- RNN

  https://zhuanlan.zhihu.com/p/34203833

    - 归纳偏置：sequentiality & time invariance
    - BPTT
    - 梯度消失与爆炸

        - 原因

          https://zhuanlan.zhihu.com/p/76772734?utm_source=wechat_session&utm_medium=social&utm_oi=46298732429312
          DNN中各个权重的梯度是独立的，该消失的就会消失，不会消失的就不会消失。
          RNN的特殊性在于，它的权重是共享的。当距离长了，最前面的导数就会消失或爆炸，但当前时刻整体的梯度并不会消失，因为它是求和的过程。
          RNN 所谓梯度消失的真正含义是，梯度被近距离梯度主导，导致模型难以学到远距离的依赖关系。

        - 解决方案

          LSTM长时记忆单元

    - LSTM

      消失：通过长时记忆单元，类似残差链接。但后来加了遗忘门，遗忘门介于0-1，梯度仍有可能消失
      爆炸：梯度仍可能爆炸，但LSTM机制复杂，多了一层激活函数sigmoid，可以通过正则与裁剪解决
      https://zhuanlan.zhihu.com/p/30465140

        - 结构
        - 各模块可以使用其他激活函数吗？

          sigmoid符合门控的物理意义
          tanh在-1到1之间，以0为中心，和大多数特征分布吻合，且在0处比sigmoid梯度大易收敛
          
          一开始没有遗忘门，也不是sigmoid，后来发现这样效果好
          
          relu的梯度是0/1，1的时候相当于同一个矩阵W连成，仍旧会梯度消失或爆炸的问题
          
          综上所述，当采用 ReLU 作为循环神经网络中隐含层的激活函数 
          时，只手言当 W的取值在单位矩阵附近时才能取得比较好的效果，因此 
          需要将 W初始化为单位矩阵。实验证明，初始化 W为单位矩阵并使用 ReLU 激活函数在一些应用中取得了与长短期记忆模型相似的结果

    - GRU

        - 结构
        - 与LSTM的异同

- Transformer

    - 结构
    - QK非对称变换

      双线性点积模型，引入非对称性，更具健壮性（Attention mask对角元素值不一定是最大的，也就是说当前位置对自身的注意力得分不一定最高）。

    - Scaled Dot Product

      为什么是缩放点积，而不是点积模型？
      当输入信息的维度 d 比较高，点积模型的值通常有比较大方差，从而导致 softmax 函数的梯度会比较小。因此，缩放点积模型可以较好地解决这一问题。
      
      相较于加性模型，点积模型具备哪些优点？
      常用的Attention机制为加性模型和点积模型，理论上加性模型和点积模型的复杂度差不多，但是点积模型在实现上可以更好地利用矩阵乘积，从而计算效率更高（实际上，随着维度d的增大，加性模型会明显好于点积模型）。

    - Multi-head

      https://zhuanlan.zhihu.com/p/76912493
      
      - 多头机制为什么有效？
      1.类似于CNN中通过多通道机制进行特征选择；
      2.Transformer中先通过切头（spilt）再分别进行Scaled Dot-Product Attention，可以使进行点积计算的维度d不大（防止梯度消失），同时缩小attention mask矩阵。

    - FFN

      Transformer在抛弃了 LSTM 结构后，FFN 中的 ReLU成为了一个主要的提供非线性变换的单元。

### 激活函数

https://zhuanlan.zhihu.com/p/73214810

- tanh

  相比Sigmoid函数，
  tanh的输出范围时(-1, 1)，解决了Sigmoid函数的不是zero-centered输出问题；
  幂运算的问题仍然存在；
  tanh导数范围在(0, 1)之间，相比sigmoid的(0, 0.25)，梯度消失（gradient vanishing）问题会得到缓解，但仍然还会存在。

    - Xavier初始化
    - 公式
    - 导数

- relu

  相比Sigmoid和tanh，ReLU摒弃了复杂的计算，提高了运算速度。
  解决了梯度消失问题，收敛速度快于Sigmoid和tanh函数
  
  缺点：
  爆炸梯度(通过梯度裁剪来解决) 
  如果学习率过大，会出现dead relu的不可逆情况 — 激活为0时不进行学习(通过加参数的ReLu解决)
  激活值的均值和方差不是0和1。(通过从激活中减去约0.5来部分解决这个问题。在fastai的视频力有个更好的解释)
  
  Leaky relu：增加了参数

    - He初始化
    - 公式
    - 导数

- gelu

  https://zhuanlan.zhihu.com/p/100175788
  https://blog.csdn.net/liruihongbob/article/details/86510622
  ReLu：缺乏随机因素，只用0和1
  
  https://www.cnblogs.com/shiyublog/p/11121839.html
  GeLu：在激活中引入了随机正则的思想，
  根据当前input大于其余inputs的概率进行随机正则化，即为在mask时依赖输入的数据分布，即x越小越有可能被mask掉，因此服从bernoulli(Φ(x))
  
  高斯误差线性单元
  对于每一个输入 x，其服从于标准正态分布 N(0, 1)，它会乘上一个伯努利分布 Bernoulli(Φ(x))，其中Φ(x) = P(X ≤ x)。
  这么选择是因为神经元的输入趋向于正太分布，这么设定使得当输入x减小的时候，输入会有一个更高的概率被dropout掉
  Gelu(x) = xΦ(x) = xP(X ≤ x)

- sigmoid

  激活函数计算量大（在正向传播和反向传播中都包含幂运算和除法）；
  反向传播求误差梯度时，求导涉及除法；
  Sigmoid的输出不是0均值（即zero-centered）；这会导致后一层的神经元将得到上一层输出的非0均值的信号作为输入，随着网络的加深，会改变数据的原始分布

    - 能否用MSE作为损失函数？
    - 公式
    - 导数
    - 优点

      激活函数计算量大（在正向传播和反向传播中都包含幂运算和除法）；
      反向传播求误差梯度时，求导涉及除法；
      Sigmoid的输出不是0均值（即zero-centered）；这会导致后一层的神经元将得到上一层输出的非0均值的信号作为输入，随着网络的加深，会改变数据的原始分布

- softmax

  sigmoid是softmax的特例：
  https://blog.csdn.net/weixin_37136725/article/details/53884173

    - 求梯度？

### 损失函数

- 分类

    - 0-1 loss
    - hinge loss
    - sigmoid loss
    - cross entropy

      求导：https://zhuanlan.zhihu.com/p/60042105

- 回归

    - square loss

      对异常点敏感

    - absolute loss

      对异常点鲁棒，但是y=f时不可导

    - Huber loss

### 优化算法

- 求解析解：凸函数
- 迭代法

    - 一阶法：梯度下降

      https://zhuanlan.zhihu.com/p/111932438

        - SGD

          数据要shuffle
          一开始重j去采用较大的学习速率 ，当 误差曲线进入平台期后，;成小学习速菜做更精细的调整。最优的学习速 率方案也通常需要调参才能得到。
          
          随机梯度下降法无法收敛
          1.bsz太小，震荡
          2.峡谷和鞍点

        - RMSProp
        - AdaGrad
        - Adam

          指数加权：
          1.不用像mean一样统计个数重新计算avg
          2.历史久远的权重会呈指数衰减
          
          动量=惯性保持：累加了之前步的速度
          1.增强了整体方向的速度，加快收敛
          2.消减了错误方向的速度，减少震荡
          
          AdaGrad=环境感知：根据不同参数的一些经验性判断 ， 自适应地确定参数的学习速率，不同参数的重新步幅 是不同的 。
          1.更新频率低的参数可以有较大幅度的更新，更新频率高的步幅可以减小。AdaGrad方法采用 “历史梯度平方和”来衡量不同参数的梯度的稀疏性 3 取值越小表明越稀疏
          参数中每个维度的更新速率都不一样！！！
          2.随着时间的推移，学习率越来越小，保证了结果的最终收敛
          
          缺点：即使Adam有自适应学习率，也需要调整整体学习率（warmup）
          
          AdamW是Adam在权重上使用了L2正则化，这样小的权重泛化性能更好。

        - LAMB

    - 二阶法：牛顿法

      在高维情况下， Hessian ~E 阵求逆的计算复杂度很大 3 而且当目标函数非口时，二阶法有可能会收 
      敛到鞍点( Saddle Point ) 。
      
      鞍点：一个不是局部最小值的驻点（一阶导数为0的点）称为鞍点。数学含义是： 目标函数在此点上的梯度（一阶导数）值为 0， 但从改点出发的一个方向是函数的极大值点，而在另一个方向是函数的极小值点。

### 正则化

- 修改数据

    - 增加数据
    - label smoothing

- 修改结构

    - Normalisation

        - Batchnorm

            - 为什么对NN层中归一化

              随着网络训练的进行 ， 每个隐层的参数变化使得后一层的输入 发生变化 3 从而每-批训练数据的分布也随之改变 3 致使网络在每次迭 代中都需要拟合不罔的数据分布，增大训练的复杂度以及过拟合的风险。

            - 为什么增加新的分布

              以Sigmoid函数为例，批量归一化 之后数据整体处于函数的非饱和区域，只包含线性变躁，破坏了之前学 习到的特征分布 。

            - 在CNN的应用

              在全连接网络中是对每个神经元进行归一化，也就是每个神经元都会学习一个γ和β
              批量归一化在卷积神经网络中应用时，需要注意卷积神经网络的 
              参数共享机制 。 每一个卷积核的参数在不同位置的楠经元当中是共享 的， 因此也应该被一起归一化。在卷积中，每层由多少个卷积核，就学习几个γ和β

            - 预测

              在预测时无法计算均值和方差，通常需要在训练时根据mini-batch和指数加权平均计算，直接用于预测

        - Layernorm

          - 对比BatchNorm
          1.对于RNN来说，sequence的长度是不一致的，换句话说RNN的深度不是固定的，不同的time-step需要保存不同的statics特征，可能存在一个特殊sequence比其他sequence长很多，这样training时，计算很麻烦。
          2.不依赖batch size
          
          在hidden size的维度进行layernorm，跟batch和seq_len无关
          beta和gamma的维度都是(hidden_size,)，每个神经元有自己的均值和方差，因为不同单元是不同的feature，量纲不一样
          
          normalisaion通常在非线性函数之前
          LN在BERT中主要起到白化的作用，增强模型稳定性（如果删除则无法收敛）

    - Dropout

      模型集成
      实现：1.训练时不动，预测时乘p 2.反向传播传播时除p，预测不动

    - weight decay

      在更新w时减去一个常数，跟L2求导之后的公式一致
      https://bbabenko.github.io/weight-decay/
      Weight decay和L2正则在SGD情况下等价，Adam下不等：https://zhuanlan.zhihu.com/p/40814046
      权重越大惩罚应该越大，但adam的adagrad调整使得惩罚变小

    - 正则项

        - L1

          稀疏解的好处：
          1.特征选择，减少计算
          2.避免过拟合，增强鲁棒性
          
          -解空间的解释：加上了菱形约束，容易在尖角处碰撞出解
          
          - 贝叶斯角度解释：加了laplace分布，在0点的概率要更高

        - L2

          -解空间角度：加了球形约束，等高线切在圆上
          -贝叶斯角度：加了高斯分布，在0点附近的概率更大且相近

- 训练技巧

    - early stopping
    - warmup

      刚开始小一些，防止对前几个batch的过拟合，之后见过了不少数据，可以慢慢升高
      之后参数基本上稳定了，就小学习率精细调整

## 统计机器学习

### 线性回归

- 
- 解析解
- 损失函数-最小二乘法

  理解：频率派角度-误差复合高斯分布的最大似然估计
  求法：
  误差服从正太分布(0,sigma) => y服从正太分布(wx,sigma)
  用高斯概率密度函数表示出y，然后进行极大似然估计

- 正则化

  从两个角度理解：
  1. 频率角度：维度太大无法求逆矩阵，且容易过拟合，给w加上约束
  X^T X是半正定，不一定可逆，X^T X + lambda I为半正定加单位矩阵，是正定的，可逆
  
  2.贝叶斯角度（最大后验）：参数符合laplace分布>L1正则，符合高斯分布>L2岭回归

    - L1
    - L2

        - 

### 线性分类

线性分类器是通过特征的线性组合来做出分类决定的分类器。
数学上来说，线性分类器能找到权值向量w，使得判别公式可以写成特征值的线性加权组合。

- 硬分类

    - 感知机

      二分类模型，y为{-1, 1}
      损失函数：误分类点到分类平面到距离，分对为0，分错>0
      L(w) = -sum(yi(wxi+b))

    - Fisher判别分析

      把样本点投影到一个平面，类间均值差大，使得类内方差小

- 软分类

  P(Y|X) = P(X|Y)P(Y) / P(X)
  
  判别模型直接求P(Y|X)
  生成模型求P(X,Y)=>P(X|Y)P(Y)=>P(Y|X)

    - 判别式

        - 逻辑回归

          由对数几率=>sigmoid：https://zhuanlan.zhihu.com/p/42656051
          公式推导：https://zhuanlan.zhihu.com/p/44591359

            - 简介

              逻辑回归是使用sigmoid作为链接函数的广义线性模型，应用于二分类任务。它假设数据服从伯努利分布，对条件概率进行建模，通过极大似然估计的方法，运用梯度下降求解参数。

                - 

            - 由来及其表达式

              用线性回归拟合 p>1-p，得到对数几率回归

            - 目标函数
            - 求解：迭代法

              为什么不求解析解？
              换成矩阵形式后，X和exp(X)同时存在，无法求出解析解。

                - 全局最优

                  逻辑回归的损失函数L是一个连续的凸函数，它只会有一个全局最优的点，不存在局部最优。可以用SGD。

            - Bias的可解释性

              对于偏差b (Bias)，一定程度代表了正负两个类别的判定的容易程度。假如b是0，那么正负类别是均匀的。如果b大于0，说明它更容易被分为正类，反之亦然。

            - 线性决策边界
            - 为什么不能用线性回归做分类？

              https://www.zhihu.com/question/319865092/answer/661614886
              平方差的意义和交叉熵的意义不一样。概率理解上，平方损失函数意味着模型的输出是以预测值为均值的高斯分布，损失函数是在这个预测分布下真实值的似然度，softmax损失意味着真实标签的似然度。

    - 生成式

        - 朴素贝叶斯

          朴素贝叶斯是基于贝叶斯定理与特征条件独立假设大分类方法，对于给定的x，对x，y的联合分布建模(P(x|y)&P(y))，输出后验概率最大的Y，对P(x|y)采用了极大似然估计
          
          当特征离散时为线性分类：
          离散特征的朴素贝叶斯分类器判别公式能够写成特征值的加权线性组合。
          https://www.jianshu.com/p/469accb2e1a0

            - 假设：条件独立性

              特征间相互独立：P(x1|y)与P(x2|y)相互独立
              P(x1, x2, .., xn | Y) = P(x1|Y) * P(x2|Y) * ... * P(xn|Y)

            - 求解

              对于给定的x，对x，y的联合分布建模(P(x|y)&P(y))，输出后验概率最大的Y，对P(x|y)采用了极大似然估计
              
              max P(x|y)P(y)，y服从伯努利分布，x|y服从categorial分布或高斯分布
              一般假设朴素贝叶斯的特征为离散值

        - 高斯判别分析

          假定已知类中的x的分别服从高斯分布，对于二分类，p(x|y=0)和p(x|y=1)分别服从两个高斯分布，方差一样，y服从bernoulli(p), P(y) = p^y(1-p)^(1-y)
          
          方差相同的情况下为线性分类（可以写成特征值x的线性加权组合）：
          https://www.jianshu.com/p/469accb2e1a0
          方差相同时把x^2消掉了，否则带有x^2就不是线性了

### SVM

https://zhuanlan.zhihu.com/p/61123737

解读：https://zhuanlan.zhihu.com/p/49331510 
考点：https://zhuanlan.zhihu.com/p/76946313

- 分类

    - 线性可分SVM

      当训练数据线性可分时，通过硬间隔(hard margin，什么是硬、软间隔下面会讲)最大化可以学习得到一个线性分类器，即硬间隔SVM

    - 线性SVM

      当训练数据不能线性可分但是可以近似线性可分时，通过软间隔(soft margin)最大化也可以学习到一个线性分类器，即软间隔SVM

    - 非线性SVM

      当训练数据线性不可分时，通过使用核技巧(kernel trick)和软间隔最大化，可以学习到一个非线性SVM

- 线性可分SVM凸二次规划形式的推导
- 拉格朗日乘子法和KKT条件
- 凸二次规划求解
- 软间隔最大化
- 序列最小优化算法(SMO)
- 核函数
- 常见问题

    - 与感知机的区别
    - 与逻辑回归的对比
    - SVM优缺点
    - 二分类到多分类

### 树模型

—XGBOOST
优点：不用做特征标准化，可以处理缺失数据，对outlier不敏感


理解泰勒展开： 
https://www.zhihu.com/question/25627482/answer/31229830
理解GBDT： 
https://www.zybuluo.com/yxd/note/611571 最好的 
https://zhuanlan.zhihu.com/p/29765582 第二好 
http://wepon.me/files/gbdt.pdf
https://www.zhihu.com/question/50121267/answer/129903947
官方文档： 
https://github.com/dmlc/xgboost/tree/master/demo
http://xgboost.readthedocs.io/en/latest/parameter.html#general-parameters
http://xgboost.readthedocs.io/en/latest/python/python_api.html
调参： 
https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
https://www.dataiku.com/learn/guide/code/python/advanced-xgboost-tuning.html
源码剖析： 
https://wenku.baidu.com/view/44778c9c312b3169a551a460.html
min_child_weight: https://www.zhihu.com/question/68621766 
scale_pos_weight: https://blog.csdn.net/h4565445654/article/details/72257538 
节点分裂：H，Weighted Quantile Sketch, h对loss有加权的作用 
稀疏值处理： 
行抽样、列抽样 
Shrinkage：学习速率减小，迭代次数增多，有正则化作用 
系统设计：Columns Block, Cache Aware Access 
Gradient-based One Side Sampling (GOSS) 
Exclusive Feature Bundling (EFB) 


*LightGBM 
官方文档： 
http://lightgbm.readthedocs.io/en/latest/
https://github.com/Microsoft/LightGBM
改进： 
直方图算法 
直方图差加速 
Leaf-wise建树 
特征并行和数据并行的优化 


* GBM 
*GBDT 
函数空间利用梯度下降法 


* Random Forest 
随机森林调参：http://www.cnblogs.com/pinard/p/6160412.html 
原理：http://www.cnblogs.com/pinard/p/6156009.html 
https://www.jianshu.com/p/dbf21ed8be88
随机森林优化： 
https://stackoverflow.com/questions/23075506/how-to-improve-randomforest-performance
树模型调参： 
https://www.zhihu.com/question/34470160/answer/114305935
https://zhuanlan.zhihu.com/p/25308120

- 信息熵相关概念
- 生成

    - ID3:信息增益
    - C4.5:信息增益比
    - CART：回归-平方误差/分类-基尼指数

- 剪枝

    - 叶节点个数
    - 预剪枝/后剪枝

- 集成

    - 随机森林
    - GBDT
    - XGBoost
    - LightGBM

### 图模型

- 有向图

  https://www.zhihu.com/question/53458773/answer/554436625
  贝叶斯网络(Bayesian Networks, BNs)是有向图, 每个节点的条件概率分布表示为P(当前节点|父节点)

    - 朴素贝叶斯

      https://www.zhihu.com/question/53458773/answer/554436625
      从朴素贝叶斯到HMM：
      在输出序列的y时，依据朴素贝叶斯只有 p(yi, xi) = P(xi|yi)P(yi)。没有考虑yi之间的关系，因此加入P(yi|yi-1)，得到HMM

    - HMM

        - 定义

          HMM是关于时序的概率模型，由一个隐藏的马尔可夫链生成不可观测的状态随机序列，再由各个状态生成观测序列

        - 三要素

          初始状态概率向量，状态转移矩阵A，观测/发射概率矩阵B

        - 假设

          齐次马尔可夫&观测独立

        - 概率计算

          给定三要素和观测序列，生成观测序列概率

        - 学习问题

          给定观测序列，用极大似然估计三要素

        - 预测/解码

          给定观测序列和三要素，求最可能的状态序列

- 无向图

  https://www.zhihu.com/question/53458773/answer/554436625
  马尔可夫网络则是无向图, 包含了一组具有马尔可夫性质的随机变量. 马尔可夫随机场(Markov Random Fields, MRF)是由参数(S,π,A)表示, 其中S是状态的集合，π是初始状态的概率, A是状态间的转移概率。一阶马尔可夫链就是假设t时刻的状态只依赖于前一时刻的状态，与其他时刻的状态和观测无关。这个性质可以用于简化概率链的计算。

    - 逻辑回归

      https://www.zhihu.com/question/265995680/answer/303148257
      朴素贝叶斯与逻辑回归的关系？
      都是对 几率P/(1-P)进行拟合：
      朴素贝叶斯基于条件独立假设，另特征间相互独立，通过P(X|Y)P(Y)=>联合概率分布求得几率
      逻辑回归拟合特征间的关系，用线性回归逼近几率

    - CRF

        - 模型定义

          举例：https://zhuanlan.zhihu.com/p/104562658
          
          无向图：在给一个节点打标签时，把相邻节点的信息考虑进来（马尔可夫性：只与相邻的两个状态有关）
          线性链条件随机场：P(Yi|X,Y1,...Yn) = P(Yi|X, Yi-1, Yi+1)，只考虑当前和前一个
          由输入序列预测输出序列的判别模型，对条件概率建模
          
          观测序列，状态/标记序列
          特征函数：转移特征t（依赖当前和前一个位置），状态特征s（依赖当前位置），t和s对取值为1或0

        - 特征函数

          转移特征t（依赖当前和前一个位置），状态特征s（依赖当前位置），t和s对取值为1或0

        - 与逻辑回归比较

          CRF是逻辑回归的序列化版本

        - 与HMM比较

          每一个HMM模型都可以用CRF构造出来
          CRF更加强大：
          1.CRF可以定义数量更多，种类更丰富的特征函数。HMM从朴素贝叶斯而来，有条件独立假设，每个观测变量只与状态变量有关。但是CRF却可以着眼于整个句子s定义更具有全局性的特征函数
          2.CRF可以使用任意的权重。将对数HMM模型看做CRF时，特征函数的权重由于是log形式的概率
          
          https://zhuanlan.zhihu.com/p/31187060
          1.HMM是生成模型，CRF是判别模型
          2.HMM是概率有向图，CRF是概率无向图
          3.HMM求解过程可能是局部最优，CRF可以全局最优（对数似然为凸函数）
          4.CRF概率归一化较合理，HMM则会导致label bias 问题

## 

