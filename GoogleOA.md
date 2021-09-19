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

\986. Interval List Intersections(range intersection)
For this range intersection problem, we need to sort the list first. While iterating the two lists, determine if the two ranges (a1, a2), (b1, b2)have common part, if they have common part, it should be(max(a1, b1), min(a2, b2)). Then we need to move the pointer who has larger end element.

```python
class Solution:
    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        firstSize, secondSize = len(firstList), len(secondList)
        i, j = 0, 0
        res = []
        while i < firstSize and j < secondSize:
            a1, a2 = firstList[i][0], firstList[i][1]
            b1, b2 = secondList[j][0], secondList[j][1]
            if a1 <= b2 and b1 <= a2:
                res.append([max(a1, b1), min(a2, b2)])
            if a2 <= b2: i += 1
            else: j += 1
        return res
```

\56. Merge Intervals (Range Intersection)
Firstly sort the list according to the first elements of the ranges.  Append the first range into the res list first, then for the incomming ranges, if the start of the incoming ranges is smaller than or equal to the end of the tail of the res list, we need to update the end of the tail of the res list to the max of these two. If the start of the incomming range is larger than the end of the tail of the res list, we need to append this incoming range into the end of the res list.

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key = lambda interval : interval[0])
        res = []
        res.append(intervals[0])
        size = len(intervals)
        for i in range(1, size):
            curInterval = intervals[i]
            start, end = curInterval[0], curInterval[1]
            curEndInRes = res[-1][1]
            if start <= curEndInRes:
                res[-1][1] = max(curEndInRes, end)
            else: res.append(curInterval)
        return res
```

range schedule problems: sort the ranges according to the end of the ranges.
\253. Meeting Rooms II

```python
# We need to keep starts and ends lists and then sort them. When come across with a start number, a meeting starts; When come across with an end number, a meeting ends. While iterating the starts list, if a start is smaller than the current end time, we need to add one to the room number, if the start is larger than or equal to the current end time, we need to add one to the end index and don't need to add one to the rooms since one meeting ends and another meeting starts, the total number of the meetings remains the same. Here is the code.
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        size = len(intervals)
        if size == 1: return 1
        starts, ends = [], []
        for interval in intervals:
            start, end = interval
            starts.append(start)
            ends.append(end)
        starts.sort()
        ends.sort()
        rooms = 0
        j = 0
        for i in range(size):
            if starts[i] < ends[j]:
                rooms += 1
            else:
                j += 1
        return rooms
            
```

```python
# For the range intersection problems, we keep a pointer for each range list. When there is a intersection between range(a1, a2) and (b1, b2), we need to put the intersection(max(a1, b1), min(a2, b2)) into the res list. Then we need to move forward the pointer the end of which index is smaller.
# For the range merge problems, we need to sort the ranges according to the start of the ranges first. Initially, set the current end to the end of the first range. While iterating the later ranges, if the start of the later range is smaller than the curren end, we can merge this range, set the current end to the max(current_end, end_of_later_range); if the start of the later range is larger than or equal to the current end, we can start a new merged range, which means to set the current end to the end of this later range.
# For the range scheduling problem(like the most unoverlapped ranges number), we can sort the ranges according to the end of the ranges. Initially set the current end to the end of the first range. For the ranges the start element of which is smaller than the current end, we can skip it. For the ranges the start element of which is larger than or equal to the current end, we can start a new range. The essential idea here is the greedy algorithm. Always select the range which ends early.
# For the hold meeting problem, we need to sort the starts and ends. When finish an end element, it represents this meeting is over. When iterating a start element, it represents a meeting starts.
# Related problems: labuladong range merge(https://github.com/labuladong/fucking-algorithm/blob/master/%E7%AE%97%E6%B3%95%E6%80%9D%E7%BB%B4%E7%B3%BB%E5%88%97/%E5%8C%BA%E9%97%B4%E8%B0%83%E5%BA%A6%E9%97%AE%E9%A2%98%E4%B9%8B%E5%8C%BA%E9%97%B4%E5%90%88%E5%B9%B6.md), range intersection(https://github.com/labuladong/fucking-algorithm/blob/master/%E7%AE%97%E6%B3%95%E6%80%9D%E7%BB%B4%E7%B3%BB%E5%88%97/%E5%8C%BA%E9%97%B4%E4%BA%A4%E9%9B%86%E9%97%AE%E9%A2%98.md), range scheduling or greedy(https://github.com/labuladong/fucking-algorithm/blob/master/%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%E7%B3%BB%E5%88%97/%E8%B4%AA%E5%BF%83%E7%AE%97%E6%B3%95%E4%B9%8B%E5%8C%BA%E9%97%B4%E8%B0%83%E5%BA%A6%E9%97%AE%E9%A2%98.md)(https://leetcode.com/discuss/interview-question/356520)
```

\435. Non-overlapping Intervals

```python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key = lambda interval : interval[1])
        currentEnd = intervals[0][1]
        size = len(intervals)
        count = 1
        for i in range(1, size):
            interval = intervals[i]
            start, end = interval
            if start >= currentEnd:
                currentEnd = end
                count += 1
        return size - count
```

\452. Minimum Number of Arrows to Burst Balloons

```python
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points.sort(key = lambda point : point[1])
        count = 1
        curEnd = points[0][1]
        size = len(points)
        for i in range(1, size):
            point = points[i]
            start, end = point
            if start > curEnd:
                curEnd = end
                count += 1
        return count
```

\973. K Closest Points to Origin

```python
# There are two ways to resolve the k smallest elements problems. 
# The first one is to heapify a list, then heappop the first k elements.
# The second one is to keep a heap of k size. While the heap size is smaller than k, use heap.heappush to push elements. While the heap size is larger than or equal to the k, use heap.heappushpop to push elements.
# Solution 1
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        temp = []
        for point in points:
            x, y = point
            dis = x ** 2 + y ** 2
            temp.append((dis, x, y))
        i = 0
        res = []
        print(temp)
        heapify(temp)
        while i < k:
            element = heapq.heappop(temp)
            print(element)
            dis, x, y = element
            res.append([x, y])
            i += 1
        return res
  # Solution 2
  class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        res = []
        for point in points:
            x, y = point
            dis = -(x ** 2 + y ** 2)
            if len(res) < k:
                heapq.heappush(res, (dis, x, y))
            else:
                heapq.heappushpop(res, (dis, x, y))
        return [[x, y] for _, x, y in res]
```

\975. Odd Even Jump

```python
# This problem is awsome!!!!  At first, I thought this problem is very similar with finding the next larger element. But after careful thinking, there is a little difference, since we are not supposed to find the next larget elements, but to find the smallest larger element in the next, so we need to sort the array [(element, index)]!To find the next smallest larger elements, we need to keep a stack to store the index. Whenever come across with an element, it is sure to larger than the previous element since this is a sorted array, we just need to check whether the index of the current element is larger than the tail of the stack, if it is, we update the nextHigher array as nextHigher[stack.pop()] = i, then push the i into the stack. After getting the nextHigher and nextLower array, we need to find whether we can jump to the end from a index. So we can keep another two array, jumpHigher and jumpLower. The end of these two array must be 1 since the end element is already the destination. To find whether we can jump higher from the current index, we just need to check whether we can jump lower from the nextHigher index. Here is the code.
class Solution:
    def oddEvenJumps(self, arr: List[int]) -> int:
        size = len(arr)
        nextHigher, nextLower = [0] * size, [0] * size
        stack = []
        for a, i in sorted([(a, i) for i, a in enumerate(arr)]):
            while stack and i > stack[-1]:
                nextHigher[stack.pop()] = i
            stack.append(i)
        stack = []
        for a, i in sorted([(-a, i) for i, a in enumerate(arr)]):
            while stack and i > stack[-1]:
                nextLower[stack.pop()] = i
            stack.append(i)
        higher, lower = [0] * size, [0] * size
        higher[-1], lower[-1] = 1, 1
        for i in range(size - 1)[::-1]:
            higher[i] = lower[nextHigher[i]]
            lower[i] = higher[nextLower[i]]
        return sum(higher)
```

\482. License Key Formatting
Think step by step.  Whenever meets with an alpha, we attach that alpha and add 1 to the count. Then we need to determine whether we should attach dash according to whether this is the first part of the string and the lenght of the first part of string equals to the modulo or the length of the current part equals to k.

```python
class Solution:
    def licenseKeyFormatting(self, s: str, k: int) -> str:
        totalCount = len(s)
        dashCount = s.count('-')
        alphaNumericCount = totalCount - dashCount
        mod = alphaNumericCount % k
        res = ""
        count = 0
        isFirstString = True
        for ch in s:
            if ch != '-':
                res += ch
                count += 1
                if (isFirstString and count == mod) or count == k:
                    if isFirstString:
                        isFirstString = False
                    res += '-'
                    count = 0
        return (res[:-1].upper())
```

\929. Unique Email Addresses

```python
class Solution:
    def numUniqueEmails(self, emails: List[str]) -> int:
        seen = set()
        for email in emails:
            local, domain = email.split('@')
            actualLocal = ''
            for ch in local:
                if ch == '.': continue
                if ch == '+': break
                actualLocal += ch
            seen.add(actualLocal + '@' +domain)
        return len(seen)
# Use the split and replace wisely  
class Solution:
    def numUniqueEmails(self, emails: List[str]) -> int:
        seen = set()
        for email in emails:
            local, domain = email.split('@')
            local = local.split('+')[0].replace('.', '')
            seen.add(local + '@' +domain)
        return len(seen)
```

\904. Fruit Into Baskets
Use sliding window to solve contiguous array related problems.

```python
class Solution:
    def totalFruit(self, fruits: List[int]) -> int:
        res = 0
        left, right = 0, 0
        typeCount = 0
        size = len(fruits)
        typeCount = dict()
        while right < size:
            fruit = fruits[right]
            typeCount[fruit] = typeCount.get(fruit, 0) + 1
            if len(typeCount) <= 2: res = max(res, right - left + 1)
            while left <= right and len(typeCount) == 3:
                leftFruit = fruits[left]
                left += 1
                typeCount[leftFruit] -= 1
                if typeCount[leftFruit] == 0:
                    del(typeCount[leftFruit])
            right += 1
        return res
```

##### Min Days to Bloom

```python
Given an array of roses. roses[i] means rose i will bloom on day roses[i]. Also given an int k, which is the minimum number of adjacent bloom roses required for a bouquet, and an int n, which is the number of bouquets we need. Return the earliest day that we can get n bouquets of roses.

Example:
Input: roses = [1, 2, 4, 9, 3, 4, 1], k = 2, n = 2
Output: 4
Explanation:
day 1: [b, n, n, n, n, n, b]
The first and the last rose bloom.

day 2: [b, b, n, n, n, n, b]
The second rose blooms. Here the first two bloom roses make a bouquet.

day 3: [b, b, n, n, b, n, b]

day 4: [b, b, b, n, b, b, b]
Here the last three bloom roses make a bouquet, meeting the required n = 2 bouquets of bloom roses. So return day 4.
```

```python
def minDaysBloom(a, k, n): 
    if not a or k == 0 or n == 0: return 0
    if len(a) < k * n: return -1
    # binary search for the left bound
    left, right = 1, max(a)
    while left <= right:
        mid = left + (right - left) // 2
        if not isValid(a, k, n, mid):
            left = mid + 1 
        else: right = mid - 1 
    return left
# count the consecutive arrays
def isValid(flowerDays, size, count, day):
    curSize, curCount = 0, 0
    for flowerDay in flowerDays:
        if flowerDay <= day:
            curSize += 1 
        else: curSize = 0
        if curSize == size:
            curCount += 1 
            curSize = 0
            if curCount == count: return True
    return False
```

\300. Longest Increasing Subsequence

```python
# Time complexity O(N2)
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        size = len(nums)
        dp = [1] * size
        for i in range(size):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)
```

```python
#Binary search to find the left bound
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        size = len(nums)
        tops = [0] * size
        piles = 0
        for num in nums:
            left, right = 0, piles - 1
            while left <= right:
                mid = left + (right - left) // 2
                if tops[mid] < num: left = mid + 1
                if tops[mid] > num: right = mid - 1
                if tops[mid] == num: right = mid - 1
            if left == piles: 
                piles += 1
            tops[left] = num
        return piles
```

\543. Diameter of Binary Tree
The recirsive return result is not the same with the required return result, we should use a helper function to return the recursive return result and update the required result during the recursive, return the required result and the main function.

```python
class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        self.res = 0
        if not root: return self.res
        self.helper(root)
        return self.res
    def helper(self, root):
        if not root: return 0
        left = self.helper(root.left)
        right = self.helper(root.right)
        self.res = max(self.res, left + right)
        return 1 + max(left, right)
```

tr ie + max diameter of binary tree https://leetcode.com/discuss/interview-question/350363/Google-or-OA-2018-or-Max-Distance

\475. Heaters
Add two dummy heaters so that each house can between two heaters.

```python
class Solution:
    def findRadius(self, houses: List[int], heaters: List[int]) -> int:
        heaters = [float('-inf')] + heaters + [float('inf')]
        heaters.sort()
        houses.sort()
        i = 0
        res = 0
        for house in houses:
            while heaters[i + 1] < house:
                i += 1
            temp = min(house - heaters[i], heaters[i + 1] - house)
            res = max(res, temp)
        return res
```

https://www.1point3acres.com/bbs/thread-797006-1-1.html

```python
# 1. find min max sum pair. To find the pairs with minimum differences, we need to sort the nums first, use two pointers at the front and the end, and move them forward each other.
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

```python
#2. get the modulo and the quotient
def find_min_diff(num):
    size = len(str(num))
    res = float('inf')
    for i in range(1, size):
        pre = num // (10 ** i)
        suffix = num % (10 ** i)
        res = min(res, abs(pre - suffix))
    return res
```

https://www.1point3acres.com/bbs/thread-798477-1-1.html

```python
# Whenever meet with the absolute value related problems, we need to remove the absolute and discuss the result according to different condition.
# It requires that |i - j| = |A[i] - A[j]|, there can be two situations. The first one is i - j = A[i] - A[j], then we can get it requires i - A[i] = j - A[j]. The second one is i - j = A[j] - A[i], it requires i + A[i] = j + A[j]. 
def max_diff(nums):
    diff_dic = dict()
    sum_dic = dict()
    left, right = 0, 0
    for i, num in enumerate(nums):
        diff = i - nums[i]
        summ = i + nums[i]
        if diff in diff_dic:
            if i - diff_dic[diff] > right - left:
                left, right = diff_dic[diff], i
        else:
            diff_dic[diff] = i
        if summ in sum_dic:
            if i - sum_dic[summ] > right - left:
                left, right = sum_dic[summ], i
        else:
            sum_dic[summ] = i
    print(diff_dic, sum_dic)
    return left, right
```

```python 
# remove b O(N2)
def min_cost(s):
    size = len(s)
    res = float('inf')
    for i in range(size):
        for j in range(i - 1, size):
            pre = s[:i]
            mid = s[i : j + 1]
            post = s[j + 1:] if j < size else ""
            print("pre:",pre)
            print("mid:",mid)
            print("post",post)
            preCount = i
            midCount = mid.count('b')
            postCount = size - 1 - j
            print(preCount, midCount, postCount)
            res = min(res, preCount + 2 * midCount + postCount)
    return res
  
```

```python
# O(N)
def countPreB(s):
    count = 0
    preCount = [0] * len(s)
    for i, ch in enumerate(s):
        if ch == 'b':
            count += 1
        preCount[i] = count
    return preCount
def min_cost(s):
    n = len(s)
    pre = countPreB(s)
    # the first and the last element are not removed
    res = 2 * pre[n - 1]
    # only remove the first element
    res = min(res, 1 - 2 * (pre[0] - pre[n - 1]))
    # only remove the last element
    if len(pre) > 1:
        res = min(res, 1 + 2 * pre[n - 2])
    dp = [0] * n
    dp[0] = 1 - 2 * pre[1]
    for i in range(1, n - 1):
        dp[i] = min(dp[i - 1], i + 1 - 2 * pre[i + 1])
    for j in range(1, n):
        localCost = n - j + 2 * pre[j] + dp[j - 1]
        res = min(res, localCost)
    return res
```













































