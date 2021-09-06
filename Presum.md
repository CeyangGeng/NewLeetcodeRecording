\560. Subarray Sum Equals K
While iterating the list, increase the current sum by adding the current element. If the current sum minus target exists in the presum dictionary, it indicates that the target exists for a sub list. Besides, we also need to add the current sum into the presum dictionary.

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        preSum = {0 : 1}
        curSum = 0
        res = 0
        for num in nums:
            curSum += num
            res += preSum.get(curSum - k, 0)
            preSum[curSum] = preSum.get(curSum, 0) + 1
        return res
```

\938. Range Sum of BST
Same idea as presum, when the low node appears, record the first presum; when the upper node appears, record the second presum, the target = second presum - first presum

```python
class Solution:
    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        self.firstSum, self.secondSum, self.curSum = 0, 0, 0
        def inorder(node):
            if not node: return 
            inorder(node.left)
            if node.val == low: 
                self.firstSum = self.curSum
            self.curSum += node.val
            if node.val == high:
                self.secondSum = self.curSum
                return
            inorder(node.right)
        inorder(root)
        return self.secondSum - self.firstSum
```

\523. Continuous Subarray Sum

```python
class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        curSum = 0
        preModSum = {0 : -1}
        for i, num in enumerate(nums):
            curSum += num
            mod = curSum % k
            if mod in preModSum.keys():
                if i - preModSum[mod] >= 2: return True
            else:
                preModSum[mod] = i
        return False
```

