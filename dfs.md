94. Binary Tree Inorder Traversal

    > Preorder: middle, left, right
    >
    > Inorder: left, middle, right
    >
    > Postorder: left, right, middle
    >
    > - recursive solution: 
    >
    >   ```python
    >   class Solution:
    >       def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    >           res = []
    >           def dfs(node):
    >               if not node: return
    >               dfs(node.left)
    >               res.append(node.val)
    >               dfs(node.right)
    >           dfs(root)
    >           return res
    >   ```
    >
    > - iterative solution
    >
    >   ```python
    >   class Solution:
    >       def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    >           stack, res = [], []
    >           while stack or root:
    >               if root: 
    >                 # find the left most node and push the parent into the stack
    >             		# we need this parent node once when we are done with the left child and need to go to the right child
    >                   stack.append(root)
    >                   root = root.left
    >               elif stack and not root:
    >                 # done with searching for the left most node, pop out the parent node
    >                   root = stack.pop()
    >                   res.append(root.val)
    >                 # go to the right child
    >                   root = root.right
    >           return res
    >   ```
    >
    >   98.  Validate Binary Search Tree
    >
    >       ```python
    >       class Solution:
    >           def isValidBST(self, root: Optional[TreeNode]) -> bool:
    >               def helper(node, lower, upper):
    >                   if not node: return True
    >                   if lower < node.val < upper: 
    >                     # update the upper and lower bound as proceeding
    >                       return helper(node.left, lower, node.val) and helper(node.right, node.val, upper)
    >                   else: return False
    >               return helper(root, float('-inf'), float('inf'))
    >       ```
    >
    >   99. Recover Binary Search Tree
    >
    >       When two elements exchange, it must be the smaller one be changed into the large position, the bigger one be changed into the small position. So if there is a non increasing pair, the first incorrect element should be the first of the abnormal pair, the second incorrect element should be the second of the abnormal pair.
    >
    >       - Recursive solution
    >
    >       ```java
    >       class Solution {
    >           public void recoverTree(TreeNode root) {
    >               inorder(root);
    >               int temp = first.val;
    >               first.val = second.val;
    >               second.val = temp;
    >           }
    >           TreeNode first = null;
    >           TreeNode second = null;
    >           TreeNode pre = null;
    >           private void inorder(TreeNode node){
    >               if (node == null) return;
    >               if (node.left != null) inorder(node.left);
    >             // this part is the operation for the inorder traverse
    >               if (pre != null && pre.val >= node.val){
    >                   if(first == null) first = pre;
    >                   second = node;
    >               }
    >               pre = node;
    >               if (node.right != null) inorder(node.right);
    >           }
    >       }
    >       ```
    >
    >       - Iterative solution
    >
    >         ```python
    >         class Solution:
    >             def recoverTree(self, root: Optional[TreeNode]) -> None:
    >                 """
    >                 Do not return anything, modify root in-place instead.
    >                 """
    >                 pre, first, second = None, None, None
    >                 stack = []
    >                 while stack or root:
    >                     if root:
    >                         stack.append(root)
    >                         root = root.left
    >                     elif stack and not root:
    >                         root = stack.pop()
    >                         # this part is the operation for the inorder traverse
    >                         if pre and pre.val >= root.val:
    >                             if not first: first = pre
    >                             second = root
    >                         pre = root
    >                         root = root.right
    >                 temp = first.val
    >                 first.val = second.val
    >                 second.val = temp
    >         ```
    >
    >       100. Same Tree
    >
    >            - Recursive Solution
    >
    >              ```python
    >              class Solution:
    >                  def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    >                      if not p and not q: return True
    >                      if p and q and p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right): return True
    >                      return False
    >              ```
    >
    >            - Iterative Solution
    >
    >              ```python
    >              class Solution:
    >                  def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    >                      stack = [(p, q)]
    >                      while stack:
    >                          first, second = stack.pop()
    >                          if first or second:
    >                              if first and not second: return False
    >                              if not first and second: return False
    >                              if first.val != second.val: return False
    >                              stack.append((first.left, second.left))
    >                              stack.append((first.right, second.right))
    >                      return True
    >              ```
    >
    >       101. Symmetric Tree
    >
    >       - Iterative solution
    >
    >         ```python
    >         class Solution:
    >             def isSymmetric(self, root: Optional[TreeNode]) -> bool:
    >                 if not root: return True
    >                 stack = [(root.left, root.right)]
    >                 while stack:
    >                     first, second = stack.pop()
    >                     if first or second: 
    >                         if first and second:
    >                             if first.val != second.val: return False
    >                             stack.append((first.left, second.right))
    >                             stack.append((first.right, second.left))
    >                         else: return False
    >                 return True
    >         ```
    >
    >       - Recursive Solution
    >
    >         ```python
    >         class Solution:
    >             def helper(self, p, q):
    >                 if not p and not q: return True
    >                 if not p or not q: return False
    >                 if p.val == q.val and self.helper(p.left, q.right) and self.helper(p.right, q.left): return True
    >                 return False
    >             def isSymmetric(self, root: Optional[TreeNode]) -> bool:
    >                 if not root: return True
    >                 return self.helper(root.left, root.right)
    >         ```
    >
    >         
    >
    >         104. Maximum Depth of Binary Tree
    >
    >              - Recursive Solution
    >
    >                ```python
    >                class Solution:
    >                    def maxDepth(self, root: Optional[TreeNode]) -> int:
    >                        if not root: return 0
    >                        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
    >                ```
    >
    >              - Iterative Solution
    >
    >                ```
    >                
    >                ```
    >
    >                
    >
    > \827. Making A Large Island
    >
    > Using color to mark different area and use map to store the area of different colors.
    >
    > Iterate over the zero elements, add surrounding island area.
    >
    > ```python
    > class Solution:
    >     def largestIsland(self, grid: List[List[int]]) -> int:
    >         colorArea = {0:0}
    >         n = len(grid)
    >         visited = set()
    >         curColor = 2
    >         for i in range(n):
    >             for j in range(n):
    >                 if grid[i][j] == 1 and (i, j) not in visited:
    >                     area = 0
    >                     stack = [(i, j)]
    >                     visited.add((i, j))
    >                     while stack:
    >                         x, y = stack.pop()
    >                         grid[x][y] = curColor
    >                         area += 1
    >                         for deltaX, deltaY in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
    >                             newX, newY = x + deltaX, y + deltaY
    >                             if 0 <= newX < n and 0 <= newY < n and grid[newX][newY] == 1 and (newX, newY) not in visited:
    >                                 stack.append((newX, newY))
    >                                 visited.add((newX, newY))
    >                     colorArea[curColor] = area
    >                     curColor += 1
    >         res = max(colorArea.values())
    >         for x in range(n):
    >             for y in range(n):
    >                 if grid[x][y] == 0:
    >                     curArea = 1
    >                     visitedColor = set()
    >                     for deltaX, deltaY in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
    >                         newX, newY = x + deltaX, y + deltaY
    >                         if 0 <= newX < n and 0 <= newY < n and grid[newX][newY] != 0 and grid[newX][newY] not in visitedColor:
    >                             visitedColor.add(grid[newX][newY])
    >                             curArea += colorArea[grid[newX][newY]]
    >                     res = max(res, curArea)
    >         return res
    > ```
    >
    > 