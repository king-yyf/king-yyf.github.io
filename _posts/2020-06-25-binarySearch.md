---
layout: post
title: 二分查找模版 
date: 2020-06-25
tags: 面试算法    
---

bug free 的二分查找模版


### 工作原理

在最简单的形式中，二分查找对具有指定左索引和右索引的连续序列进行操作。这就是所谓的查找空间。二分查找维护查找空间的左、右和中间指示符，并比较查找目标或将查找条件应用于集合的中间值；如果条件不满足或值不相等，则清除目标不可能存在的那一半，并在剩下的一半上继续查找，直到成功为止。如果查以空的一半结束，则无法满足条件，并且无法找到目标。  


在接下来的章节中，我们将回顾如何识别二分查找问题，“为什么我们使用二分查找” 这一问题的原因，以及你以前可能不知道的 3 个不同的二分查找模板。由于二分查找是一个常见的面试主题，我们还将练习问题按不同的模板进行分类，以便你可以在实践使用到每一个。  

**注意**  

二进制搜索可以采用许多替代形式，并且可能并不总是直接搜索特定值。有时您希望应用特定条件或规则来确定接下来要搜索的哪一侧（左侧或右侧）。


### 模版一

模板 #1 是二分查找的最基础和最基本的形式。这是一个标准的二分查找模板，大多数高中或大学会在他们第一次教学生计算机科学时使用。模板 #1 用于查找可以通过访问数组中的单个索引来确定的元素或条件。

**关键属性**

>* 二分查找的最基础和最基本的形式。
>* 查找条件可以在不与元素的两侧进行比较的情况下确定（或使用它周围的特定元素）。
>* 不需要后处理，因为每一步中，你都在检查是否找到了元素。如果到达末尾，则知道未找到该元素。

**区分语法**  

>* 初始条件：`left = 0, right = length-1`
>* 终止：`left > right`
>* 向左查找：`right = mid-1`
>* 向右查找：`left = mid+1`


  c++模版：  
  ```c++
  int binarySearch(vector<int>& nums, int target){
    if(nums.size() == 0)
      return -1;
    int left = 0, right = nums.size() - 1;
    while(left <= right){
      //Prevent (left + right) overflow
      int mid = left + (right - left) / 2;
      if(nums[mid] == target){
        return mid;
      }
      else if(nums[mid] < target){
        left = mid + 1;
      }else{
        right = mid - 1;
      }
    }
    // End Condition: left > right
    return -1;
  }
  ```
  Java模版：
  ```java
  int binarySearch(int[] nums, int target){
  if(nums == null || nums.length == 0)
    return -1;

  int left = 0, right = nums.length - 1;
  while(left <= right){
    // Prevent (left + right) overflow
    int mid = left + (right - left) / 2;
    if(nums[mid] == target){ return mid; }
    else if(nums[mid] < target) { left = mid + 1; }
    else { right = mid - 1; }
  }

  // End Condition: left > right
  return -1;
}
  ```

  Python模版：
  ```python
  def binarySearch(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    if len(nums) == 0:
        return -1

    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    # End Condition: left > right
    return -1
  ```
  
### 模版二

模板 #2 是二分查找的高级模板。它用于查找需要访问数组中当前索引及其直接右邻居索引的元素或条件。  

**关键属性**

>* 一种实现二分查找的高级方法。
>* 查找条件需要访问元素的直接右邻居。
>* 使用元素的右邻居来确定是否满足条件，并决定是向左还是向右。
>* 保证查找空间在每一步中至少有 2 个元素。
>* 需要进行后处理。 当你剩下 1 个元素时，循环 / 递归结束。 需要评估剩余元素是否符合条件。

**区分语法**  

>* 初始条件：`left = 0, right = length`
>* 终止：`left == right`
>* 向左查找：`right = mid`
>* 向右查找：`left = mid+1`

  c++模版：
  ```c++
  int binarySearch(vector<int>& nums, int target){
    if(nums.size() == 0) return -1;

    int left = 0, right = nums.size();
    while(left < right){
      int mid = left + (right - left) / 2;
      if(nums[mid] == target){ return mid;}
      else if(nums[mid] < target){ left = mid + 1;}
      else{
        right = mid;
      }
    }

    // Post-processing
    // End Condition: left == right
    if(left != nums.size() && nums[left] == target)
      return left;

    return -1;
  }
  ```

  Java模版：
  ```java
  int binarySearch(int[] nums, int target){
  if(nums == null || nums.length == 0)
    return -1;

  int left = 0, right = nums.length;
  while(left < right){
    // Prevent (left + right) overflow
    int mid = left + (right - left) / 2;
    if(nums[mid] == target){ return mid; }
    else if(nums[mid] < target) { left = mid + 1; }
    else { right = mid; }
  }

  // Post-processing:
  // End Condition: left == right
  if(left != nums.length && nums[left] == target) return left;
  return -1;
}
  ```

  Python模板：
  ```python
  def binarySearch(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    if len(nums) == 0:
        return -1

    left, right = 0, len(nums)
    while left < right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid

    # Post-processing:
    # End Condition: left == right
    if left != len(nums) and nums[left] == target:
        return left
    return -1
  ```

### 模版三

模板 #3 是二分查找的另一种独特形式。 它用于搜索需要访问当前索引及其在数组中的直接左右邻居索引的元素或条件。

**关键属性**  

>* 实现二分查找的另一种方法。
>* 搜索条件需要访问元素的直接左右邻居。
>* 使用元素的邻居来确定它是向右还是向左。
>* 保证查找空间在每个步骤中至少有 3 个元素。
>* 需要进行后处理。 当剩下 2 个元素时，循环 / 递归结束。 需要评估其余元素是否符合条件。


**区分语法**  

>* 初始条件：`left = 0, right = length-1`
>* 终止：`left + 1 == right`
>* 向左查找：`right = mid`
>* 向右查找：`left = mid`

  c++代码：
  ```c++
  int binarySearch(vector<int> & nums, int target){
    if(nums.size() == 0) return -1;

    int left = 0, right = nums.size() - 1;
    while (left + 1 < right){
        // Prevent (left + right) overflow
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) {
            return mid;
        } else if (nums[mid] < target) {
            left = mid;
        } else {
            right = mid;
        }
    }
    // Post-processing:
    // End Condition: left + 1 == right
    if(nums[left] == target) return left;
    if(nums[right] == target) return right;
    return -1;
  }
  ```

  Java代码:
  ```java
  int binarySearch(int[] nums, int target) {
    if (nums == null || nums.length == 0)
        return -1;

    int left = 0, right = nums.length - 1;
    while (left + 1 < right){
        // Prevent (left + right) overflow
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) {
            return mid;
        } else if (nums[mid] < target) {
            left = mid;
        } else {
            right = mid;
        }
    }

    // Post-processing:
    // End Condition: left + 1 == right
    if(nums[left] == target) return left;
    if(nums[right] == target) return right;
    return -1;
}
  ```

  Python代码:
  ```python
  def binarySearch(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    if len(nums) == 0:
        return -1

    left, right = 0, len(nums) - 1
    while left + 1 < right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid
        else:
            right = mid

    # Post-processing:
    # End Condition: left + 1 == right
    if nums[left] == target: return left
    if nums[right] == target: return right
    return -1
  ```

