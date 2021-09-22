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

\1249. Minimum Remove to Make Valid Parentheses

```python
# Use stack to store the index of the invalid parenthesis. When encountering with the left part, put the index of it into the stack. When encountering with the right part, check whether the top of stack is the right part. If it is, pop out this right part; if it isn't, put this invalid right part into the stack.
class Solution:
    def minRemoveToMakeValid(self, s: str) -> str:
        stack = []
        s = list(s)
        for i, ch in enumerate(s):
            if ch == '(':
                stack.append(i)
            elif ch == ')':
                if stack and s[stack[-1]] == '(':
                    stack.pop()
                else:
                    stack.append(i)
        while stack:
            s[stack.pop()] = ''
        return ''.join(s)
```

\953. Verifying an Alien Dictionary

```python
# Use zip to form pairs. Iterating the two word in zip pairs. For the characters at the same index of the first word and the second word, if the first character is smaller than the second character, then we can return true; if the first character is larger than the second character, then we can return false; only when these two characters are the same, we need to proceed. If we still can not tell which one is larger, it shows that one of the word is starts with the other. Then we only need to compare the length of these two words. If the first word is longer, we need to return False, otherwise we need to return True.
class Solution:
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        pairs = list(zip(words, words[1:]))
        dic = dict()
        for i, ch in enumerate(order):
            dic[ch] = i
        for pair in pairs:
            if not self.isValidPair(pair, dic):
                return False
        return True
    def isValidPair(self, pair, dic):
        first, second = pair
        size1, size2 = len(first), len(second)
        for i in range(min(size1, size2)):
            c1, c2 = first[i], second[i]
            if dic[c1] < dic[c2]: 
                return True
            if dic[c1] > dic[c2]:
                return False
        if size1 > size2: return False
        return True
```

\680. Valid Palindrome II

```python
class Solution:
    def validPalindrome(self, s: str) -> bool:
        left, right = 0, len(s) - 1
        while left < right:
            if s[left] != s[right]:
                temp1 = s[left : right]
                temp2 = s[left + 1: right + 1]
                return temp1 == temp1[::-1] or temp2 == temp2[::-1]
            left += 1
            right -= 1
        return True
```

\1762. Buildings With an Ocean View

```python
# Monotonic stack
class Solution:
    def findBuildings(self, heights: List[int]) -> List[int]:
        res = []
        size = len(heights)
        stack = []
        for i in range(size - 1, -1, -1):
            height = heights[i]
            while stack and height > stack[-1]:
                stack.pop()
            if stack: continue
            res.append(i)
            stack.append(height)
        return res[::-1]
```

\1570. Dot Product of Two Sparse Vectors

```python
# Use dictionary to record the index and the number. While iterating the key value pair in one vector, we need to check whether this key also exists in the self dictionary. If it doesn't, we don't need to do the production and sum.
class SparseVector:
    def __init__(self, nums: List[int]):
        self.dic = dict()
        for i, num in enumerate(nums):
            self.dic[i] = num
        

    # Return the dotProduct of two sparse vectors
    def dotProduct(self, vec: 'SparseVector') -> int:
        res = 0
        for key, val in vec.dic.items():
            if key in self.dic:
                res += val * self.dic[key]
        return res
```

\973. K Closest Points to Origin

```python
# Use heapq
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        heap = []
        for x, y in points:
            dis = -(x ** 2 + y ** 2)
            if len(heap) < k:
                heapq.heappush(heap, (dis, x, y))
            else:
                heapq.heappushpop(heap, (dis, x, y))
        return [[x, y] for _, x, y in heap]
```

\426. Convert Binary Search Tree to Sorted Doubly Linked List

```python
class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        if not root: return None
        leftHead = self.treeToDoublyList(root.left)
        rightHead = self.treeToDoublyList(root.right)
        root.left = root
        root.right = root
        return self.connect(self.connect(leftHead, root), rightHead)
    def connect(self, n1, n2):
        if not n1: return n2
        if not n2: return n1
        tail1 = n1.left
        tail2 = n2.left
        tail1.right = n2
        n2.left = tail1
        tail2.right = n1
        n1.left = tail2
        return n1
```

\415. Add Strings

```python
# Keep a carry
class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        num1, num2 = list(num1), list(num2)
        add = 0
        if len(num1) > len(num2): num1, num2 = num2, num1
        size = len(num1)
        size2 = len(num2)
        res = ""
        for i in range(-1, -size - 1, -1):
            n1, n2 = int(num1[i]), int(num2[i])
            temp = n1 + n2 + add
            remain = temp % 10
            quotient = temp // 10
            add = quotient
            res = str(remain) + res
        if len(num2) > len(num1):
            for i in range(-size - 1, -size2 - 1, -1):
                n = int(num2[i])
                temp = n + add
                quotient = temp // 10
                remain = temp % 10
                add = quotient
                res = str(remain) + res
        if add != 0:
            res = str(add) + res
        return res
```

\301. Remove Invalid Parentheses

```python
# Use count to determine when there is one more right part.
# After finding the position making the count negative, we can remove any right part before this position, then the remain string should be deal with recursive. 
# To avoid duplication, we need to keep two more information, the first one is the last_i which makes the count negative, and the i in the new iteration starts from i; the second one is the last_j to avoid remove two parenthesis just in different order.
# After finding the position making the count negative, we just need to iterate the j from last_j to i(inclusive). After finishing iterating the j, we can return since the i in the later position making count negative is left to recursive to deal with.

class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        res = []
        def helper(last_i, last_j, string, parenthesis, res):
            count = 0
            for i in range(last_i, len(string)):
                if string[i] == parenthesis[0]:
                    count += 1
                elif string[i] == parenthesis[1]:
                    count -= 1
                if count >= 0: continue
                for j in range(last_j, i + 1):
                    if string[j] == parenthesis[1] and (j == last_j or (j > last_j and string[j] != string[j - 1])):
                        newString = string[:j] + string[j + 1:]
                        helper(i, j, newString, parenthesis, res)
                return
            reverse = string[::-1]
            if parenthesis[0] == '(':
                parenthesis = [')', '(']
                helper(0, 0, reverse, parenthesis, res)
            else:
                res.append(reverse)
        helper(0, 0, s, ['(', ')'], res)
        return res
```

\199. Binary Tree Right Side View

```python
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        if not root: return res
        queue = [root]
        while queue:
            res.append(queue[-1].val)
            temp = []
            for node in queue:
                if node.left: temp.append(node.left)
                if node.right: temp.append(node.right)
            queue = temp
        return res
```

\636. Exclusive Time of Functions
Add the interval to each of the start timestamp.

```python
class Solution:
    def exclusiveTime(self, n: int, logs: List[str]) -> List[int]:
        dic = dict()
        stack = []
        for log in logs:
            funcID, event, timeStamp = log.split(":")
            funcID = int(funcID)
            timeStamp = int(timeStamp)
            if event == "start":
                stack.append(timeStamp)
            elif event == "end":
                dic[funcID] = dic.get(funcID, 0)
                duration = timeStamp - stack.pop() + 1
                dic[funcID] += duration
                temp = []
                for time in stack:
                    temp.append(time + duration)
                stack = temp
                    
        res = []
        for i in range(len(dic)):
            res.append(dic[i])
        return res
```

\236. Lowest Common Ancestor of a Binary Tree
Record the parent of each node, then find the ancestors of p iteratively, find the ancestors of q iteratively, find if they have common ancestors.

```python

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        dic = dict()
        dic[root] = None
        stack = []
        stack.append(root)
        while p not in dic or q not in dic:
            node = stack.pop()
            if node.left: 
                dic[node.left] = node
                stack.append(node.left)
            if node.right: 
                dic[node.right] = node
                stack.append(node.right)
        parents = set()
        while p:
            parents.add(p)
            p = dic[p]
        while q not in parents:
            q = dic[q]
        return q
        
```

\238. Product of Array Except Self
Compute the pre product and post product.

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        pre = [1]
        size = len(nums)
        for i in range(size - 1):
            num = nums[i]
            pre.append(pre[-1] * num)
        post = nums[-1]
        for i in range(size - 2, -1, -1):
            pre[i] *= post
            post *= nums[i]
        return pre
```

\721. Accounts Merge
Use the common email as connection to construct graph. Construct the emails_accounts_dictionary first. Then for each account, use each email to find the neighbor. Always add element to visited at first visit.

```python
class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        size = len(accounts)
        visited_accounts = [False] * size
        emails_accounts_map = defaultdict(list)
        for i, account in enumerate(accounts):
            for j in range(1, len(account)):
                email = account[j]
                emails_accounts_map[email].append(i)
        def dfs(i, emails):
            account = accounts[i]
            for j in range(1, len(account)):
                email = account[j]
                emails.add(email)
                for neighbor in emails_accounts_map[email]:
                    if not visited_accounts[neighbor]:
                        visited_accounts[neighbor] = True
                        dfs(neighbor, emails)
        res  = []
        for i, account in enumerate(accounts):
            if visited_accounts[i]: continue
            visited_accounts[i] = True
            emails = set()
            name = account[0]
            dfs(i, emails)
            res.append([name] + sorted(list(emails)))
        return res
```

\543. Diameter of Binary Tree
The recursive variable is different from what the problem ask for, so we need a helper function. In the main function, maintain a global variable to be returned, update this global variable in the helper function. The return value of the helper function should be the longest path but not the diameter. Another key point is that for the node as a root, the return of helper(node.left) = longest path in the left node(including), the return of helper(node.right) = longest path in right. When calculating the diameter for the root, it should be left + right, we don't need to add another one.

```python
class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        self.d = 0
        if not root: return self.d
        left = self.helper(root.left)
        right = self.helper(root.right)
        return max(self.d, left + right)
    def helper(self, node):
        if not node: return 0
        left, right = 0, 0
        if node.left: left = self.helper(node.left)
        if node.right : right = self.helper(node.right)
        self.d = max(self.d, left + right)
        return max(left, right) + 1
```

\227. Basic Calculator II
The basic idea is keep two variable sign and number, when come across with numeric string, add it to number, when come across with sign, calculate the sign and number. If the sign is + or -, pretty easy. If the sign is '*', we need to pop out the number, and multiply it with the number. If the sign is '/', we need to pop out the previous number, if the previous number is negative and the modulo of the previous number divide current number is not zero, we need to add one to quotient of previous number dividing current number. Another key point is, there are severl if condition, the first if is to determine whether the character is numeric, the second if is to determine whether the current character is non-numeric, then we need to calculate the result now, but another very tricky place is that if we only rely on the sign to hint the calculation, the final caculation can not be guaranteed. So in the non-numeric if condition, we need to or with the i == len(s) - 1. But before calculation, we need to gurantee that this final character string is been considered, so the first two conditions are if and if, but not if and elif. Another tricky place is that since we need to check whether i is the last index of the string, so we need to use s.strip to remove the leading and ending blank space, the code is s = s.strip().

```python
class Solution:
    def calculate(self, s: str) -> int:
        s = s.strip()
        sign, num = '+', 0
        stack = []
        for i, ch in enumerate(s):
            if ch == ' ': continue
            if ch.isnumeric():
                num = num * 10 + int(ch)
            if (not ch.isnumeric()) or i == len(s) - 1 :
                if sign == '+': stack.append(num)
                elif sign == '-': stack.append(-num)
                elif sign == '*': stack.append(num * stack.pop())
                elif sign == '/': 
                    pre = stack.pop()
                    if pre < 0 and pre % num != 0: 
                        stack.append(pre // num + 1)
                    else: stack.append(pre // num)
                
                sign = ch
                num = 0
        return sum(stack)
```

\670. Maximum Swap
To get the maximum number with only one swap, we need to find out the later larger index and iterate from left to right. If the current number is the largest one in the numbers behind it self(including itself), we don't need to swap for this index and proceed. To find out the later larger index, we need to iterate from the tail to the head. The larger index of the last number is the index of itself. One corner case is that if there are multiple same largest number in the later positions, we need to choose the right most one. E.g., 1993, for the number 1 at the index 0, we need to swap it with the second nine but not the first nine, so we need to update the postLargerValueIndex only when the current number is larger thatn all of the numbers behind it.  After getting the later larger index, we need to iterate from the left to the right, if the current number is equal to the later larger number, we skip this position until we find out the current number is smaller than the later larger number.

```python
class Solution:
    def maximumSwap(self, num: int) -> int:
        nums = []
        while num:
            mod = num % 10
            num = num // 10
            nums = [mod] + nums
        size = len(nums)
        laterMaxValueIndex = []
        laterMaxValueIndex.append(size - 1)
        postMaxValueIndex = size - 1
        for i in range(size - 2, -1, -1):
            curNum = nums[i]
            if curNum > nums[postMaxValueIndex]: postMaxValueIndex = i
            laterMaxValueIndex = [postMaxValueIndex] + laterMaxValueIndex
        res = 0
        print(laterMaxValueIndex)
        for i, num in enumerate(laterMaxValueIndex):
            first, second = nums[i], nums[num]
            if first == second: continue
            else: 
                nums[i], nums[num] = nums[num], nums[i]
                break
        print(nums)
        for num in nums:
            res = res * 10 + num
        return res
```

\249. Group Shifted Strings
Since "ba" is in the same group with "az", we need to use (26 + ord(string[i]) - ord(string[i - 1])) % 26 to find the members in the same group. Another key point is that to concact element to the current tuple and get a nuew tuple, we need to use tuple += (element, )

```python
class Solution:
    def groupStrings(self, strings: List[str]) -> List[List[str]]:
        dic = defaultdict(list)
        for string in strings:
            key = ()
            for i in range(1, len(string)):
                temp = (26 + (ord(string[i]) - ord(string[i - 1]))) % 26
                key += (temp, )
            dic[key].append(string)
        return list(dic.values())
```

\140. Word Break II
Basic idea for word break is to concact the pre valid word to the left behind words list. The basic case is when the string is empty, we need to return list with empty string but not an empty list. 

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        if not s: return ['']
        wordDict = set(wordDict)
        memo = {}
        size = len(s)
        res = []
        for i in range(size):
            pre = s[:i + 1]
            validRight = []
            if pre in wordDict:
                validRight = self.wordBreak(s[i + 1:], wordDict)
            for valid in validRight:
                if not valid: res.append(pre)
                else: res.append(pre + ' ' + valid)
        return res
```

```python
# Use memo to avoid duplicate calculation
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        self.wordDict = wordDict
        self. memo = {}
        return self.helper(s)
    def helper(self, s):
        if not s: return ['']
        if s in self.memo: return self.memo[s]
        res = []
        for i in range(len(s)):
            word = s[:i + 1]
            if word in self.wordDict:
                leftValidWord = self.helper(s[i + 1:])
                for valid in leftValidWord:
                    if not valid: res.append(word)
                    else: res.append(word + " " + valid)
        self.memo[s] = res
        return res
```

\67. Add Binary
Make up 0 for the length difference.

```python
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        if len(a) < len(b): a, b = b, a
        lenDiff = len(a) - len(b)
        b = "0" * lenDiff + b
        carrier = 0
        res = ""
        for i in range(-1, -len(a) - 1, -1):
            first, second = int(a[i]), int(b[i])
            remain = (first + second + carrier) % 2
            carrier = (first + second + carrier) // 2
            res = str(remain) + res
        if carrier > 0:
            res = str(carrier) + res
        return res
```



































