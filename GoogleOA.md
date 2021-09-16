\1509. Minimum Difference Between Largest and Smallest Value in Three Moves

```python
class Solution:
    def minDifference(self, nums: List[int]) -> int:
        nums = sorted(nums)
        size = len(nums)
        if size <= 3: return 0
        minDiff = float('inf')
        for pair in zip(nums[size - 4 :], nums[:4]):
            first, second = pair
            minDiff = min(minDiff, first - second)
        return minDiff
```

\1525. Number of Good Ways to Split a String
Keep two dictionary to record the unique characters before this index and after this index. Then iterate the string length to find where the number of the unique character of the formmer string and latter string are the same.

```python
class Solution:
    def numSplits(self, s: str) -> int:
        pre = dict()
        suffix = dict()
        preSeen = set()
        suffixSeen = set()
        preUnique = 0
        suffixUnique = 0
        for i, ch in enumerate(s):
            if ch in preSeen:
                pre[i] = preUnique
                continue
            preUnique += 1
            pre[i] = preUnique
            preSeen.add(ch)
        for i, ch in enumerate(s[::-1]):
            if ch in suffixSeen:
                suffix[i] = suffixUnique
                continue
            suffixUnique += 1
            suffix[i] = suffixUnique
            suffixSeen.add(ch)
        size = len(s)
        res = 0
        for i in range(size - 1):
            if pre[i] == suffix[size - 2 - i]:
                res += 1
        return res
```

\949. Largest Time for Given Digits
Iterate over all the permutations using python library: itertools.permutations

```python
class Solution:
    def largestTimeFromDigits(self, arr: List[int]) -> str:
        res = ""
        for p in permutations(arr):
            if p[0] * 10 + p[1] <= 23 and p[2] <= 5:
                res = max(res, str(p[0]) + str(p[1]) + ":" + str(p[2]) + str(p[3]))
        return res
```

Maximum Time
You are given a string that represents time in the format `hh:mm`. Some of the digits are blank (represented by `?`). Fill in `?` such that the time represented by this string is the maximum possible. Maximum time: `23:59`, minimum time: `00:00`. You can assume that input string is always valid.

``` python
class Solution:
  def maxTime(self, s: str) -> str:
    res = list(s)
    if res[0] == '?':
      res[0] = '2' if res[1] < '4' or res[1] == '?' else '1'
    if res[1] == '?':
      res[1] = '3' if res[0] == '2' else '9'
    if res[2] == '?':
    	res[2] = '5'
    if res[3] == '?':
      res[3] = '9'
    return ''.join(res)
```

Given a list, getting all the possible sum from this list, we can use this code:

```python
def calculateAllPossibleSum(l: list):
  sums = {0}l;koplkmkop c
  for number in l:
    sums |= {number + i for i in sums}
  return sums
```

\1049. Last Stone Weight II
The essence of this question is to find the smallest difference between two subarrays

```python
def minDiff(stones):
  s = set(0)
  for stone in stones:
    s |= {stone + i for i in s}
  res = float('inf')
  total = sum(stones)
  for group in s:
    res = min(res, abs(total - 2 * group))
  return res
```

##### Google | OA 2019 | Most Booked Hotel Room

Given a hotel which has 10 floors `[0-9]` and each floor has 26 rooms `[A-Z]`. You are given a sequence of rooms, where `+` suggests room is booked, `-` room is freed. You have to find which room is booked maximum number of times.You may assume that the list describe a correct sequence of bookings in chronological order; that is, only free rooms can be booked and only booked rooms can be freeed. All rooms are initially free. Note that this does not mean that all rooms have to be free at the end. In case, 2 rooms have been booked the same number of times, return the lexographically smaller room.

- N (length of input) is an integer within the range [1, 600]
- each element of array A is a string consisting of three characters: "+" or "-"; a digit "0"-"9"; and uppercase English letter "A" - "Z"
- the sequence is correct. That is every booked room was previously free and every freed room was previously booked.

```
Input: ["+1A", "+3E", "-1A", "+4F", "+1A", "-3E"]
Output: "1A"
Explanation: 1A as it has been booked 2 times.
```

```python
def max_booking(bookings):
  dic = dict()
  for booking in bookings:
    if booking[0] == '-': continue
    room = booking[1:]
    dic[room] = dic.get(room, 0) + 1
  res = ''
  max_frequency = 0
  for room, frequency in dic.items():
    if frequency > max_frequency: res = room
    if frequency == max_frequency: res = min(res, room)
  return res
```

\1007. Minimum Domino Rotations For Equal Row
Index -> element -> transform element into index -> element.
When the elements are limited, we can use the number as index, the list value as the number count.

```python
class Solution:
    def minDominoRotations(self, tops: List[int], bottoms: List[int]) -> int:
        countTop, countBottom, same = [0] * 7, [0] * 7, [0] * 7
        size = len(tops)
        for i in range(size):
            countTop[tops[i]] += 1
            countBottom[bottoms[i]] += 1
            if tops[i] == bottoms[i]:
                same[tops[i]] += 1
        for i in range(1, 7):
            if countTop[i] + countBottom[i] - same[i] == size:
                return min(countTop[i], countBottom[i]) - same[i]
        return -1
```

\1165. Single-Row Keyboard
Use dictionary.

```python
class Solution:
    def calculateTime(self, keyboard: str, word: str) -> int:
        dic = dict()
        for i, ch in enumerate(keyboard):
            dic[ch] = i
        res = 0
        prePos = 0
        for w in word:
            curPos = dic[w]
            res += abs(curPos - prePos)
            prePos = curPos
        return res
```

\1161. Maximum Level Sum of a Binary Tree
BFS

```python
class Solution:
    def maxLevelSum(self, root: Optional[TreeNode]) -> int:
        queue = [root]
        res, maxSum, curLevel = 1, root.val, 1
        while queue:
            temp = []
            levelSum = 0
            for node in queue:
                levelSum += node.val
                if node.left: temp.append(node.left)
                if node.right: temp.append(node.right)
            queue = temp
            if levelSum > maxSum:
                maxSum = levelSum
                res = curLevel
            print(levelSum)
            curLevel += 1
        return res
```

monotonic stack: To solve problems with the monotonic stack skill, we need to maintain a stack which keep a decrease numbers. We also need to iterate from the end of the array, if the current number is larger than the top of the stack, we need to pop out the top element since this top element is smaller than the current number, we want to get the number which is larger than the current number. So we need to stop popping when meeting with a larger element. As a result, there will be a decrease array in this stack.
monotonic queue: To solve problems with the monotonic queue skill, we need to maintain a queue which keeps a decrease array. We also need to iterate the array from the beginning. If the current number is larger than the end of the queue, we need to pop out the end of the queue, then push it to the end.
\239. Sliding Window Maximum

```python
# Every time adding a new number, determine if the curren index - head of the queue exceeds the k, if it is, pop the head first. # Then comparing the current number with the end of the queue, while the current number is larger than the tail, we need to pop 
# it out. In this way, we keep a decrease array in this queue. Here is the code.
# Always pop the the number that is smaller than the current number from the end of the queue. 
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        queue, res = [], []
        for i, num  in enumerate(nums):
            if queue and i - queue[0] == k:
                queue.pop(0)
            while queue and num > nums[queue[-1]]:
                queue.pop()
            queue.append(i)
            res.append(nums[queue[0]])
        return res[k - 1:]

```

\496. Next Greater Element I

```python
# Use monotonic stack, always pop out the number smaller than the current number from the tail of the stack. In the end, there will be a decreasing array in the stack.
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        dic = dict()
        stack = []
        for num in nums2[::-1]:
            while stack and num > stack[-1]:
                stack.pop()
            if not stack: dic[num] = -1
            else: dic[num] = stack[-1]
            stack.append(num)
        res = []
        for num in nums1:
            res.append(dic[num])
        return res
```

\503. Next Greater Element II
Use module to deal with circular array.

```python
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        size = len(nums)
        stack = []
        res = [0] * size
        for i in range(2 * size - 1, -1, -1):
            while stack and stack[-1] <= nums[i % size]:
                stack.pop()
            res[i % size] = stack[-1] if stack else -1
            stack.append(nums[i % size])
        return res
```

\739. Daily Temperatures
Use monotonic stack technique.

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        size = len(temperatures)
        stack, res = [], [0] * size
        for i in range(size - 1, -1, -1):
            temperature = temperatures[i]
            while stack and temperatures[stack[-1]] <= temperature:
                stack.pop()
            if stack:
                res[i] = stack[-1] - i
            stack.append(i)
        return res
```































Ω