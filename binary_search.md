1. Find a number

   ```python
   def binary_search(nums, target):
     left, right = 0, len(nums) - 1
     while (left <= right):
       mid = left + (right - left) / 2
       if nums[mid] < target: left = mid + 1
       if nums[mid] > target: right = mid - 1
       if nums[mid] == target: return mid
     return -1
   ```

2. Find the left bound

   ```python
   def left_bound(nums, target):
     left, right = 0, len(nums) - 1
     while (left <= right):
       mid = left + (right - left) / 2
       if nums[mid] < target: left = mid + 1
       if nums[mid] > target: right = mid - 1
       if nums[mid] == target: right = mid - 1
     if left == len(nums) or nums[left] != target: return -1
     return left
   ```

3. Find the right bound

   ```python
   def right_bound(nums, target):
     left, right = 0, len(nums) - 1
     while (left <= right):
       mid = left + (right - left) / 2
       if nums[mid] < target: left = mid + 1
       if nums[mid] > target: right = mid - 1
       if nums[mid] == target: left = mid + 1
     if right < 0 or nums[right] != target: return -1
     return right
   ```

   \528. Random Pick with Weight

   This problem is to find the left border of the elements that are larger than or equal to the target.
   Another key point is that for the random.randint (start, end), the start and end are both included.

   ```python
   class Solution:
   
       def __init__(self, w: List[int]):
           self.nums = list()
           cur = 0
           for n in w:
               cur += n
               self.nums.append(cur)
           self.end = self.nums[-1]
   
       def pickIndex(self) -> int:
           target = random.randint(1, self.end)
           left, right = 0, len(self.nums) - 1
           # find the left bound of elements that are larger than or equal to the target 
           while left <= right:
               mid = left + (right - left) // 2
               if self.nums[mid] < target: left = mid + 1
               if self.nums[mid] > target: right = mid - 1
               if self.nums[mid] == target: right = mid - 1
           return left
   ```

   