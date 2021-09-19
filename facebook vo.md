1.\125. Valid Palindrome
Move two pointers towards each other. Initial the left pointer to be 0, and initial the right pointer to be length - 1. The condition of the while loop should be while i < j, instead of i <= j since if i == j, we don't need to check it these two characters are the same. Another problem is that if the while conditionis i <= j and we add 1 to the i or decrease 1 to the j when the character is not alnumeric, the i or j might exceeds the length of the string,

```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        size = len(s)
        i, j = 0, size - 1
        while i < j:
            while i < j and not s[i].isalnum():
                i += 1
            while i< j and not s[j].isalnum():
                j -= 1
            if s[i].lower() != s[j].lower():
                return False
            i += 1
            j -= 1
        return True
```

\5. Longest Palindromic Substring
Extend the string from the center. If the final string is of odd length, we the left center and right center should be the same; If the final string is of even length, the left center and right center should have difference of exactlly one.

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        size = len(s)
        if size < 2: return s
        res = ""
        for i in range(size - 1):
            odd = self.extend(s, i, i)
            even = self.extend(s, i, i + 1)
            temp = odd if len(odd) > len(even) else even
            if len(temp) > len(res):
                res = temp
        return res
    def extend(self, string, left, right):
        size = len(string)
        while left >= 0 and right < size:
            if string[left] != string[right]:
                break
            left -= 1
            right += 1
        return string[left + 1 : right]
```

\1293. Shortest Path in a Grid with Obstacles Elimination
Use BFS to solve the shortest path in grid related problems.  There are two keys in the BFS, the first one is seen set to avoid repeated searches. The second one is queue to iterating all the elements. The elements in the queue should store all the status information. For this problem, we need to store the position information, the left eliminates and the steps. The seen set should store the position and the left eliminates, if this has been seen, we don't need to track this path any more.

```python
class Solution:
    def shortestPath(self, grid: List[List[int]], k: int) -> int:
        if len(grid) == 1 and len(grid[0]) == 1: return 0
        m, n = len(grid), len(grid[0])
        queue, seen = [], set()
        queue.append((0, 0, k, 0))
        seen.add((0, 0, k))
        while queue:
            x, y, leftEliminate, steps = queue.pop(0)
            if leftEliminate < 0: continue
            if x == m - 1 and y == n - 1: return steps
            for deltaX, deltaY in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                newX, newY = x + deltaX, y + deltaY
                if 0 <= newX < m and 0 <= newY < n:
                    if grid[newX][newY] == 1 and (newX, newY, leftEliminate - 1) not in seen:
                        seen.add((newX, newY, leftEliminate - 1))
                        queue.append((newX, newY, leftEliminate - 1, steps + 1))
                    elif grid[newX][newY] == 0 and (newX, newY, leftEliminate) not in seen:
                        seen.add((newX, newY, leftEliminate))
                        queue.append((newX, newY,leftEliminate, steps + 1))
                
        return -1
```

\314. Binary Tree Vertical Order Traversal

```python
# Use bfs to gurantee the elements in each list is from top to the bottom. Use x index to gurantee the the element on the same vertical line are in the same list. We also need to record the xmin and xmax while iterating.
class Solution:
    def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root: return []
        dic = dict()
        queue = [(0, root)]
        xMin, xMax = 0, 0
        while queue:
            x, node = queue.pop(0)
            dic[x] = dic.get(x, list())
            dic[x].append(node.val)
            xMin = min(x, xMin)
            xMax = max(x, xMax)
            if node.left:
                queue.append((x - 1, node.left))
            if node.right:
                queue.append((x + 1, node.right))
        res = []
        for i in range(xMin, xMax + 1):
            res.append(dic[i])
        return res
```

\227. Basic Calculator II

```python
# Use stack to deal with calculator related problems. The basic idea is to deal with operator with higher priority. When meet with a number, check if the end of the stack is also a number, if it is, then pop the end number, and combine it with the new number, push the new combined number into the end of the stack; If the end of the stack is a operator, we need to check the previous operator, if the previous operator is with higher priority, we need to deal with this higher priority operation first, then put the result into the end of the stack. After finish iterating the string, we also need to check whether the last operator is of higher priority, if it is, then compute the result, and put the result back to the stack. After this process, there will only be plus and minus operations in the stack, so we can just pop out two elements one time, if the operator is plus, then add the number to the final result, if the operator is minus, then minus the number to the final result.
class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        for ch in s:
            if ch.isnumeric():
                if stack and stack[-1] not in {'+', '-', '*', '/'}:
                    lastNum = stack.pop()
                    lastNum = lastNum * 10 + int(ch)
                    stack.append(lastNum)
                else:
                    stack.append(int(ch))
            else:
                if ch == ' ': continue
                if len(stack) > 2 and stack[-2] in {'*', '/'}:
                    second, op, first = stack.pop(), stack.pop(), stack.pop()
                    temp = self.evaluation(first, op, second)
                    stack.append(temp)
                stack.append(ch)
        print(stack)
        if len(stack) > 2 and stack[-2] in {'*', '/'}:
            second, op, first = stack.pop(), stack.pop(), stack.pop()
            stack.append(self.evaluation(first, op, second))
        stack = ['+'] + stack
        res = 0
        while stack:
            operator, number = stack.pop(0), stack.pop(0)
            if operator == '+': res += number
            else: res -= number
        return res
    
    def evaluation(self, first, op, second):
        if op == '+': return first + second
        if op == '-': return first - second
        if op == '*': return first * second
        if op == '/': return first // second
     
```

```python
# Another thought is to keep two in air variables, the operator and the number. Whenever meet a numerical number, add this numberical number to the number. If(noting this is if but not elif) it is not a numerical numer, and it is not a space or the current index is the last index, we need to combine the operator and the number. The divide case is very tricky, if the stack.pop() // number < 0 and stack.pop() % number != 0, we need to add one to the // result.
class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        sign, num = '+', 0
        for i, ch in enumerate(s):
            if ch.isnumeric():
                num = num * 10 + int(ch)
            if (not ch.isnumeric() and ch != ' ') or i == len(s) - 1:
                if sign == '+': stack.append(num)
                if sign == '-': stack.append(-num)
                if sign == '*': stack.append(stack.pop() * num)
                if sign == '/': 
                    temp = stack.pop()
                    if temp // num < 0 and temp % num != 0:
                        stack.append(temp // num + 1)
                    else:stack.append(temp // num)
                sign = ch
                num = 0
        return sum(stack)
```

\938. Range Sum of BST
Similar with presum problem. While using the inorder iterating, if the current node value equals to the low value, record the current sum; If the node value equals to the high value, record the current value and return this function. The difference of these two sum is the range sum.

```python
class Solution:
    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        self.firstSum, self.secondSum, self.curSum = 0, 0, 0
        self.inorder(root, low, high)
        return self.secondSum - self.firstSum
    def inorder(self, root, low, high):
        if not root: return
        self.inorder(root.left, low, high)
        if (root.val == low):
            self.firstSum = self.curSum
        self.curSum += root.val
        if (root.val == high):
            self.secondSum = self.curSum
            return
        self.inorder(root.right, low, high)
```

\791. Custom Sort String

```python
# The most strait forward method is to iterating the characters in the order. plus the result with s.count(ch) * ch. After finishing iterating the order string, we need to iterate the s string to add the characters that don't exist in the order.
class Solution:
    def customSortString(self, order: str, s: str) -> str:
        res = ""
        for ch in order:
            if ch in s:
                res += ch * s.count(ch)
        for ch in s:
            if ch not in order:
                res += ch
        return res
```

\31. Next Permutation

```python
# 1. Find the longest non increasing suffix.
# 2. Find the pivot.
# 3. If the pivot not exist, then reverse the nums.
# 4. If the pivot exists, find the last element that is larger than the number right before the pivot.
# 5. Reverse the suffix right after the pivot(including the pivot).
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        size = len(nums)
        pivot = float('inf')
        for i in range(size - 1, 0, -1):
            if nums[i] > nums[i - 1]:
                pivot = i
                break
        if pivot == float("inf"):
            nums.reverse()
        else:
            for j in range(size - 1, pivot - 1, -1):
                if nums[j] > nums[pivot - 1]:
                    nums[j], nums[pivot - 1] = nums[pivot - 1], nums[j]
                    break
            left, right = pivot, size - 1
            while left < right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1
```

![image-20210918131258531](/Users/gengceyang/Library/Application Support/typora-user-images/image-20210918131258531.png)

```python
def findMinMaxPair(A):
  A.sort()
  res = float('-inf')
  size = len(A)
  i, j = 0, size - 1
  while i < j:
    curSum = A[i] + A[j]
    if curSum > res:
      res = curSum
  return res
```

![image-20210918131540524](/Users/gengceyang/Library/Application Support/typora-user-images/image-20210918131540524.png)

```python
def find_min_diff(num):
    size = len(str(num))
    res = float('inf')
    for i in range(1, size):
        pre = num // (10 ** i)
        suffix = num % (10 ** i)
        res = min(res, abs(pre - suffix))
    return res
```

















