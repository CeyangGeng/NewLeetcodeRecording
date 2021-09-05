\973. K Closest Points to Origin
the heap in python keep the top k largest elements. The heapq.pop() will pop out the smallest element. There are several heapq functions: heappush, heap pop, heapify, heappushpop.

```python
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        h = []
        for x, y in points:
            dist = - (x ** 2 + y ** 2)
            if len(h) == k:
                heapq.heappushpop(h, (dist, x, y))
            else:
                heapq.heappush(h, (dist, x, y))
        res = []
        return [[x, y] for _, x, y in h]
```

