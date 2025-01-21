'''
CHALLENGES INDEX

20. Valid Parentheses (Stack)
23. Merge k Sorted Lists (LL) (DQ) (Heap) (Sorting)
42. Trapping Rain Water (Array) (TP) (DS) (Stack)
150. Evaluate Reverse Polish Notation (Stack)
155. Min Stack (Stack)
215. Kth Largest Element in an Array (Array) (Heap) (DQ) (Sorting)
218. The Skyline Problem (Heaps) (DQ)
227. Basic Calculator II (Stack)
230. Kth Smallest Element in a BST (Heap) (DFS) (Tree)
295. Find Median from Data Stream (Heap) (Sorting)
347. Top K Frequent Elements (Array) (Heaps) (Sorting)
378. Kth Smallest Element in a Sorted Matrix (Matrix) (Heaps)

32. Longest Valid Parentheses (Stack) (DP)
394. Decode String (RC) (Stack)
739. Daily Temperatures (Array) (Stack) [Monotonic Stack]
71. Simplify Path (Stack)


*LL: Linked-Lists
*BS: Binary Search
*DP: Dynamic Programming
*RC: Recursion
*TP: Two-pointers
*FCD: Floyd's cycle detection (Hare & Tortoise approach)
*PS: Preffix-sum
*SW: Sliding-Window
*MEM: Memoization
*GRE: Greedy
*DQ: Divide and Conquer
*BT: Backtracking
*BFS & DFS: Breadth-First Search & Depth-First Search
*Arrays, Hash Tables & Matrices
*Sorting
*Heaps, Stacks & Queues
*Graphs & Trees
*Others

(16)
'''


'20. Valid Parentheses'
# def x():

#     from typing import Optional

#     # Input
#     # Case 1
#     s = "()"
#     # Output: True

#     # Case 2
#     s = "()[]{}"
#     # Output: True

#     # Case 3
#     s = "(]"
#     # Output: False

#     # Case 4
#     s = "([])"
#     # Output: True

#     # Custom Case
#     s = "))"
#     # Output: False

#     '''
#     My Approach (Stack)

#         Intuition:
            
#             - Hanlde corner case: If input string length is not even
#             - Initialize a 'stack' holder to store the closing parenthesis.
#             - Initialize a 'par' dictionary at each opening parentesis char as key and its correspondent closing as value.
#             - Iterate from right to left:
#                 + Pop each element.
#                 + if the element is a closing parenthesis inserted as first value in 'stack'.
#                 + else:
#                     + The first 'stack' element should correspond (with 'par's help) to the last popped, if it doesn't:
#                         * Return False
#                         * Otherwise, pop the first element of 'stack' and continue the iterations.
            
#             - If the code gets to this point, return True.

#     '''

#     def isValid(s: str) -> bool:
        
#         # Turn the input into a list
#         s = list(s)

#         # Handle Corner case: Odd lengthed input
#         if len(s)%2 != 0:
#             return False
        
#         # Initialize a 'stack' holder to store the closing parenthesis.
#         stack = []

#         # Initialize a 'par' dictionary
#         par = {
#             '(':')',
#             '{':'}',
#             '[':']',
#         }

#         for i in range(len(s)-1,-1,-1):

#             char = s[i]

#             if char in ')}]':
#                 stack.insert(0, char)
            
#             else:

#                 if not stack or par[char] != stack[0]:
#                     return False

#                 else:
#                     stack.pop(0)

#         # Return True if it gets to this point
#         return True if not stack else False

#     # Testing
#     print(isValid(s=s))

#     'Notes: it works!'

'23. Merge k Sorted Lists'
# def x():

#     # Base
#     class ListNode(object):
#         def __init__(self, val=0, next=None):
#             self.val = val
#             self.next = next

#     # Input

#     # 1st Input
#     #List 1
#     one1, two1, three1 = ListNode(1), ListNode(4), ListNode(5)
#     one1.next, two1.next = two1, three1

#     #List 2
#     one2, two2, three2 = ListNode(1), ListNode(3), ListNode(4)
#     one2.next, two2.next = two2, three2

#     #List 3
#     one3, two3 = ListNode(2), ListNode(6)
#     one3.next = two3

#     # List of lists
#     li = [one1, one2, one3]

#     # My Approach

#     '''
#     Rationale:
    
#         1. Create an empty node.
#         2. Assign the node with the minimum value as next
#         3. Move that node to its next node until reaches 'None'.
#         4. When every value within the input list is None, breakout the loop and return.
#     '''

#     def mergeKLists(lists:list[ListNode]) -> ListNode:
        
#         lists = [x for x in lists if x.val != '']

#         if len(lists) == 0:
#             return ListNode('')


#         head = ListNode('')
#         curr = head
#         li = lists

#         while True:

#             if li == [None]:
#                 break

#             # Create a list of the current nodes in input that aren't None and sort them ascendingly by value
#             li = sorted([node for node in li if node != None], key = lambda x: x.val)

#             # Make the 'next_node' the next node to the curr None & move over to that node right away
#             curr.next = li[0]
#             curr = curr.next

#             # Move over to the next node of next_node
#             li[0] = li[0].next

#         return head.next

#     # Testing
#     res = mergeKLists([ListNode('')])
#     res_li = []

#     print(res)

#     'Notes: It worked'

'42. Trapping Rain Water'
# def x():

#     # Input

#     # case 1
#     height = [0,1,0,2,1,0,1,3,2,1,2,1]  # Exp. Out: 6

#     # case 2
#     height = [4,2,0,3,2,5]  # Exp. Out: 9


#     'Solution'
#     def trap(height):

#         if not height:
#             return 0
        

#         left, right = 0, len(height)-1
#         left_max, right_max = 0, 0
#         result = 0

#         while left < right:

#             if height[left] < height[right]:

#                 if height[left] >= left_max:
#                     left_max = height[left]

#                 else:
#                     result += left_max - height[left]

#                 left += 1
            
#             else:

#                 if height[right] >= right_max:
#                     right_max = height[right]

#                 else:
#                     result += right_max - height[right]

#                 right -= 1
        
#         return result

#     # Testing
#     print(trap([3,0,2]))

#     'Done'

'''150. Evaluate Reverse Polish Notation'''
# def x():

#     # Input
#     # Case1
#     tokens = ["2","1","+","3","*"]
#     #Output: 9 / ((2 + 1) * 3)

#     # Case2
#     tokens = ["4","13","5","/","+"]
#     #Output: 6 / (4 + (13 / 5) )

#     # Case3
#     tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
#     #Output: ((((((9 + 3) * -11) / 6) * 10) + 17) + 5)
    

#     '''
#     My approach

#         Intuition

#             Look from left to right the first operand and operate with the trailing two elements (which should be digits)
#             and return the result to where the first of the digits were and iterate the same process until ther is only
#             3 elements in the list to make the last operation.
#     '''

#     def evalPRN(tokens:list[str]) -> int:    

#         operations = [x for x in tokens if x in '+-*/']

#         for _ in range(len(operations)):

#             operation = operations.pop(0)
#             idx = tokens.index(operation)
#             return_idx = idx-2

#             operand, num2, num1 = tokens.pop(idx), tokens.pop(idx-1), tokens.pop(idx-2)

#             if operand == '/':

#                 num1, num2 = int(num1), int(num2)
#                 op_result = num1//num2 if num2 > 0 else (num1 + (-num1 % num2)) // num2

#             else:
#                 op_result = eval(num1+operand+num2)

#             tokens.insert(return_idx, str(op_result))
            
#         return tokens[0]

#     # Testing
#     print(evalPRN(tokens=tokens))

#     'Note: My solution worked for 81% of the cases'


#     'Another Approach'
#     def evalRPN(tokens):
            
#         stack = []

#         for x in tokens:

#             if stack == []:
#                 stack.append(int(x))

#             elif x not in '+-/*':
#                 stack.append(int(x))

#             else:

#                 l = len(stack) - 2

#                 if x == '+':
#                     stack[l] = stack[l] + stack.pop()

#                 elif x == '-':
#                     stack[l] = stack[l] - stack.pop()

#                 elif x == '*':
#                     stack[l] = stack[l] * stack.pop()

#                 else:
#                     stack[l] = float(stack[l]) / float(stack.pop())
#                     stack[l] = int(stack[l])    

#         return stack[0]

#     # Testing
#     print(evalRPN(tokens=tokens))

#     'Done'

'''155. Min Stack'''
# def x():

#     # Input
#     # Case 1
#     commands = ["MinStack","push","push","push","getMin","pop","top","getMin"]
#     inputs = [[],[-2],[0],[-3],[],[],[],[]]
#     # Output: [None,None,None,None,-3,None,0,-2]

#     # Custom Case
#     commands = ["MinStack","push","push","push","top","pop","getMin","pop","getMin","pop","push","top","getMin","push","top","getMin","pop","getMin"]
#     inputs = [[],[2147483646],[2147483646],[2147483647],[],[],[],[],[],[],[2147483647],[],[],[-2147483648],[],[],[],[]]
#     # Output: [None,None,None,None,-3,None,0,-2]


#     'Solution'
#     class MinStack(object):

#         def __init__(self):
#             self.stack = []
#             self.min = None
            

#         def push(self, val):
#             """
#             :type val: int
#             :rtype: None
#             """
#             self.stack.append(val)

#             if not self.min:
#                 self.min = val
            
#             else:
#                 self.min = min(val, self.min)
            

#         def pop(self):
#             """
#             :rtype: None
#             """
#             item = self.stack.pop()

#             if item == self.min:
                
#                 self.min = min(self.stack) if self.stack else None


#         def top(self):
#             """
#             :rtype: int
#             """
#             return self.stack[-1]
            

#         def getMin(self):
#             """
#             :rtype: int
#             """
#             return self.min
            

#     # Testing
#     for i, command in enumerate(commands):

#         if command == 'MinStack':
#             stack = MinStack()
        
#         elif command == 'push':
#             stack.push(inputs[i][0])   

#         elif command == 'pop':
#             stack.pop()    
        
#         elif command == 'top':
#             res = stack.top()

#         elif command == 'getMin':
#             res = stack.getMin()

#     'Note: My solution worked for 97% of the cases'


#     'Another solution'
#     class MinStack(object):

#         def __init__(self):
#             self.stack = []
                    

#         def push(self, val):
#             """
#             :type val: int
#             :rtype: None
#             """

#             if not self.stack:
#                 self.stack.append([val, val])
#                 return
            
#             min_elem = self.stack[-1][1]

#             self.stack.append([val, min(val, min_elem)])
            

#         def pop(self):
#             """
#             :rtype: None
#             """
#             self.stack.pop()
            

#         def top(self):
#             """
#             :rtype: int
#             """
#             return self.stack[-1][0]
            

#         def getMin(self):
#             """
#             :rtype: int
#             """
#             return self.stack[-1][1]

#     'Done'

'''215. Kth Largest Element in an Array'''
# def x():

#     'Solution'
#     import heapq

#     def findKthLargest(self, nums: list[int], k: int) -> int:
#             heap = nums[:k]
#             heapq.heapify(heap)
            
#             for num in nums[k:]:
#                 if num > heap[0]:
#                     heapq.heappop(heap)
#                     heapq.heappush(heap, num)
            
#             return heap[0]

#     'Done'

'''218. The Skyline Problem'''
# def x():

#     '''
#     Explanation of the Code

#         Events Creation:

#             For each building, two events are created: entering ((left, -height, right)) and exiting ((right, height, 0)).
        
#         Sorting Events:

#             Events are sorted first by x-coordinate. If x-coordinates are the same, entering events are processed before exiting events. For entering events with the same x-coordinate, taller buildings are processed first.
        
#         Processing Events:

#             A max-heap (live_heap) keeps track of the current active buildings' heights. Heights are stored as negative values to use Python's min-heap as a max-heap.
#             When processing each event, heights are added to or removed from the heap as needed.
#             If the maximum height changes (top of the heap), a key point is added to the result.
        
#         This approach efficiently manages the skyline problem by leveraging sorting and a max-heap to dynamically track the highest building at each critical point.
#     '''

#     from heapq import heappush, heappop, heapify

#     def getSkyline(buildings: list[list[int]]) -> list[list[int]]:
            
#         # Create events for entering and exiting each building
#         events = []

#         for left, right, height in buildings:
#             events.append((left, -height, right))  # Entering event
#             events.append((right, height, 0))     # Exiting event
        

#         # Sort events: primarily by x coordinate, then by height
#         events.sort()
        

#         # Max-heap to store the current active buildings
#         result = []
#         live_heap = [(0, float('inf'))]  # (height, end)


#         # Process each event
#         for x, h, r in events:

#             if h < 0:  # Entering event
#                 heappush(live_heap, (h, r))

#             else:  # Exiting event
                
#                 # Remove the building height from the heap
#                 for i in range(len(live_heap)):
#                     if live_heap[i][1] == x:
#                         live_heap[i] = live_heap[-1]  # Replace with last element
#                         live_heap.pop()  # Remove last element
#                         heapify(live_heap)  # Restore heap property
#                         break
            
#             # Ensure the heap is valid
#             while live_heap[0][1] <= x:
#                 heappop(live_heap)
            
#             # Get the current maximum height
#             max_height = -live_heap[0][0]
            
#             # If the current maximum height changes, add the key point
#             if not result or result[-1][1] != max_height:
#                 result.append([x, max_height])
                    
#         return result

#     'Done'

'''227. Basic Calculator II'''
# def x():

#     # Input
#     # Case 1
#     s = "3+2*2"
#     # Output: 7

#     # Case 2
#     s = " 3/2 "
#     # Output: 1

#     # Case 3
#     s = " 3+5 / 2 "
#     # Output: 5

#     # Custom Case
#     s = "1+2*5/3+6/4*2"
#     # Output: 5


#     '''
#     My Approach

#         Intuition:

#             1. Process the string to make valid expression elements.
#             2. Process each operator:
#                 - '/*-+' in that order, until there is none left.
#                 - Take each operator and the element to the left and to the right to compose a new element to insert it 
#                     where the left one where.
#             3. Return the result.
#     '''

#     def calculate(s: str) -> int:
        
#         # Handle no operators case
#         if not any(op in s for op in '/*-+'):
#             return int(s)
        

#         # Process the String to make it a valid Expression List
#         expression = []
#         num = ''

#         for char in s:

#             if char != ' ':

#                 if char in '+-*/':
#                     expression.append(num)
#                     expression.append(char)
#                     num = ''
                
#                 else:
#                     num += char

#         expression.append(num)  # Append the last number in the string


#         # Process the '*' and the '/' in the expression list until there are no more of those operators
#         while any(op in expression for op in '*/'):

#             for elem in expression:

#                 if elem == '*':
#                     idx = expression.index('*')
#                     new_element = int(expression[idx-1]) * int(expression[idx+1])
#                     expression = expression[:idx-1] + [new_element] + expression[idx+2:]
                
#                 elif elem == '/':
#                     idx = expression.index('/')
#                     new_element = int(expression[idx-1]) // int(expression[idx+1])
#                     expression = expression[:idx-1] + [new_element] + expression[idx+2:]

        
#         # Process the '+' and the '-' in the expression list until there are no more of those operators
#         while any(op in expression for op in '+-'):

#             for elem in expression:
                                            
#                 if elem == '+':
#                     idx = expression.index('+')
#                     new_element = int(expression[idx-1]) + int(expression[idx+1])
#                     expression = expression[:idx-1] + [new_element] + expression[idx+2:]
                
#                 elif elem == '-':
#                     idx = expression.index('-')
#                     new_element = int(expression[idx-1]) - int(expression[idx+1])
#                     expression = expression[:idx-1] + [new_element] + expression[idx+2:]


#         # Return the result
#         return expression[0]

#     # Testing
#     print(calculate(s=s))

#     '''
#     Notes: 
#         This approach met 97% of the cases and it only breaks by time-limit.
#     '''


#     'Stack Approach'
#     import math

#     def calculate(s:str) -> int:

#         num = 0
#         pre_sign = '+'
#         stack = []

#         for char in s+'+':

#             if char.isdigit():
#                 num = num*10 + int(char)

#             elif char in '/*-+':

#                 if pre_sign == '+':
#                     stack.append(num)
                
#                 elif pre_sign == '-':
#                     stack.append(-num)
                            
#                 elif pre_sign == '*':
#                     stack.append(stack.pop()*num)            
                
#                 elif pre_sign == '/':
#                     stack.append(math.trunc(stack.pop()/num))
                
#                 pre_sign = char
#                 num = 0
        
#         return sum(stack)

#     print(calculate(s=s))

#     'Done'

'''230. Kth Smallest Element in a BST'''
# def x():

#     # Base
#     # Definition for a binary tree node.
#     class TreeNode:
#         def __init__(self, val=0, left=None, right=None):
#             self.val = val
#             self.left = left
#             self.right = right

#     # Case 1
#     tree_layout = [3,1,4,None,2]
#     one, four = TreeNode(val=1, right=TreeNode(val=2)), TreeNode(val=4)
#     root = TreeNode(val=3, left=one, right=four)
#     k = 1
#     # Output: 1

#     # Case 2
#     tree_layout = [5,3,6,2,4,None,None,1]
#     three, six = TreeNode(val=3, left=TreeNode(val=2, left=TreeNode(val=1)), right=TreeNode(val=4)), TreeNode(val=6)
#     root = TreeNode(val=5, left=three, right=six)
#     k = 3
#     # Output: 3

#     # Custom Case
#     tree_layout = [5,3,6,2,4,None,None,1]
#     three, six = TreeNode(val=3, left=TreeNode(val=2, left=TreeNode(val=1)), right=TreeNode(val=4)), TreeNode(val=6)
#     root = TreeNode(val=5, left=three, right=six)
#     k = 3
#     # Output: 3


#     '''
#     My Aprroach
   
#         Intuition:
#             - Traverse the Tree with preorder to extract the values
#             - Create a Max heap of length k and go through the rest of the elements (mantaining the heap property).
#             - Return the first element of the heap.
#     '''

#     def kth_smallest(root: TreeNode,k: int) -> int:

#         # Define Aux Inorder traversal func
#         def inorder(root: TreeNode, path:list) -> list:

#             if root:

#                 node = root

#                 inorder(root=node.left, path=path)
#                 path.append(node.val)
#                 inorder(root=node.right, path=path)

#                 return path

#         tree_list = inorder(root=root, path=[])

#         tree_list.sort()

#         return tree_list[k-1]

#     # Testing
#     print(kth_smallest(root=root, k=k))

#     '''Notes: 
#     - This approach works perfectly, and it beated 37% of solutions in Runtime and 80% in space.
        
#         Complexity:
#         - Time complexity: O(nlogn).
#         - Space Complexity: O(n).

#     Now, if no sorting func is required to be used, below will be that version.
#     '''


#     'Without Sorting Approach'
#     import heapq

#     def kth_smallest(root: TreeNode,k: int) -> int:

#         # Define Aux Inorder traversal func
#         def inorder(root: TreeNode, path:list) -> list:

#             if root:

#                 node = root

#                 inorder(root=node.left, path=path)
#                 path.append(node.val)
#                 inorder(root=node.right, path=path)

#                 return path

#         # Extract the tree nodes values in a list
#         tree_list = inorder(root=root, path=[])


#         # Make a min-heap out of the tree_list up to the 'k' limit
#         heap = tree_list[:k]
#         heapq.heapify(heap)

#         # Iterate through each element in the tree_list starting from 'k' up to len(tree_list)
#         for num in tree_list[k:]:

#             if num < heap[0]:
#                 heapq.heappop(heap)
#                 heapq.heappush(heap, num)
        
#         return heap[-1] # The result is the last element of the min-heap, since it was length k, and the last is the kth

#     # Testing
#     print(kth_smallest(root=root, k=k))

#     '''Notes: 
#     - This approach also worked smoothly, and it consequentially reduced its performance
#         beating only 6% of solutions in Runtime and it maintains the 80% in space.
        
#         Complexity:
#         - Time complexity: O(n+(n-k)logk).
#         - Space Complexity: O(n).

#     Now, what if I don't traverse the elements (O(n)) and later I traverse up to k?
#         Would it be possible to order the heap while traversing the tree?.
#     '''

#     'Another enhanced solution'
#     import heapq

#     def kth_smallest(root: TreeNode, k: int) -> int:

#         # Define the heap with 'inf' as it first element (To be pushed later on)
#         heap = [float('inf')]

#         # Define Aux Inorder traversal func
#         def inorder(root: TreeNode) -> None:

#             if root:

#                 node = root

#                 inorder(root=node.left)

#                 if len(heap) == k:

#                     if node.val < heap[0]:
#                         heapq.heappop(heap)
#                         heapq.heappush(heap, node.val)
#                         pass
                
#                 else:
#                     heap.append(node.val)


#                 inorder(root=node.right)
        
#         inorder(root=root)
        
#         return heap[-1] # The result is the last element of the min-heap, since it was length k, and the last is the kth

#     # Testing
#     print(kth_smallest(root=root, k=k))

#     '''Notes: 
#     - This approach also worked smoothly, and it actually beated the first approach in performance,
#         beating 57% of solutions in Runtime and it maintains the 80% in space.
        
#         Complexity:
#         - Time complexity: O(nlogk).
#         - Space Complexity: O(n+k).

#     That was a great exercise, now what is the customary solution for this?.
#         Quick answer: Simply inorderlt traverse the tree up to k, since is a Binary Search Tree, it was already sorted.
#     '''

#     'Done'

'''295. Find Median from Data Stream'''
# def x():

#     # Input
#     # Case 1
#     commands = ["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
#     inputs = [[], 1, 2, [], 3, []]
#     # Output: [None, None, None, 1.5, None, 2.0]

#     # Case 2
#     commands = ["MedianFinder","addNum","findMedian","addNum","findMedian","addNum","findMedian","addNum","findMedian","addNum","findMedian"]
#     inputs = [[],-1,[],-2,[],-3,[],-4,[],-5,[]]
#     # Output: [None, None, -1.0, None, -1.5, None, -2.0, None, -2.5, None, -3.0]


#     'My approach'
#     import heapq

#     class MedianFinder:

#         def __init__(self):
#             self.nums = []      

#         def addNum(self, num: int) -> None:

#             # heapq.heappush(self.nums, num)
#             # heapq.heapify(self.nums)

#             self.nums.append(num)
#             self.nums.sort()
            

#         def findMedian(self) -> float:

#             nums_len = len(self.nums)

#             if nums_len % 2 == 0:
#                 mid1, mid2 = nums_len//2-1, nums_len//2
#                 return (self.nums[mid1]+self.nums[mid2])/2
            
#             else:            
#                 mid = nums_len//2
#                 return self.nums[mid]

#     # Testing
#     obj = MedianFinder()

#     for idx, command in enumerate(commands):

#         if command == 'addNum':
#             print(obj.addNum(inputs[idx]))
        
#         elif command == 'findMedian':
#             print(obj.findMedian())


#     'Note: This solution met 95% of cases but sorting in every addition cauases inefficiencies'


#     'Heaps approach'
#     import heapq

#     class MedianFinder:

#         def __init__(self):
#             self.small = []  # Max-heap (inverted values)
#             self.large = []  # Min-heap

#         def addNum(self, num: int) -> None:

#             # Add to max-heap (invert to simulate max-heap)
#             heapq.heappush(self.small, -num)
            
#             # Balance the heaps
#             if self.small and self.large and (-self.small[0] > self.large[0]):
#                 heapq.heappush(self.large, -heapq.heappop(self.small))
            
#             # Ensure the sizes of the heaps differ by at most 1
#             if len(self.small) > len(self.large) + 1:
#                 heapq.heappush(self.large, -heapq.heappop(self.small))

#             if len(self.large) > len(self.small):
#                 heapq.heappush(self.small, -heapq.heappop(self.large))

#         def findMedian(self) -> float:

#             # If the heaps are of equal size, median is the average of the tops
#             if len(self.small) == len(self.large):
#                 return (-self.small[0] + self.large[0]) / 2
            
#             # Otherwise, the median is the top of the max-heap
#             return -self.small[0]

#     'Done'

'''347. Top K Frequent Elements'''
# def x():

#     # Input
#     # Case 1
#     nums = [1,2,2,1]
#     k = 2
#     # Output: [1,2]

#     # Case 2
#     nums = [1]
#     k = 1
#     # Output: [1]


#     '''
#     My approach

#         Intuition:
            
#         Ideas' pool:
#             + A Counter function approach:
#                 - Call a Counter on the input and sort by freq, return in order.

#             + A Heap approach:
#                 - ...    
#     '''

#     def topKFrequent(nums: list[int], k: int) -> list[int]:

#         # Create the result list holder
#         result = []

#         # Import Counter
#         from collections import Counter

#         #  Transform the input into a list sorted by freq
#         nums = sorted(Counter(nums).items(), key=lambda x: x[1], reverse=True)

#         # Populate the result accordingly
#         for i in range(k):
#             result.append(nums[i][0])

#         # Return the result
#         return result

#     # Testing
#     print(topKFrequent(nums=nums, k=k))

#     'Note: This approach worked beating submissions only 20% in Runtime and 61% in Memory'

#     'Done'

'''378. Kth Smallest Element in a Sorted Matrix'''
# def x():

#     # Input
#     # Case 1
#     matrix = [[1,5,9],[10,11,13],[12,13,15]]
#     k = 8
#     # Output: 13 / The elements in the matrix are [1,5,9,10,11,12,13,13,15], and the 8th smallest number is 13

#     # Case 2
#     matrix = matrix = [[-5]]
#     k = 1
#     # Output: -5


#     '''
#     My Approach

#         Intuition:

#             Ideas' pool:

#                 + Brute forcing: flatten the input, sort and return.
#                 + Heap: Hold a min heap of size k, traverse all items in the matrix and return the last of the heap.

#     '''

#     'Brute force'
#     def kthSmallest(matrix: list[list[int]], k: int) -> int:

#         # Flatten the input
#         matrix = [x for elem in matrix for x in elem]

#         # Sort the resulting matrix
#         matrix.sort()

#         # x=0 

#         # Return the kth element
#         return matrix[k-1]

#     # Testing
#     print(kthSmallest(matrix=matrix, k=k))

#     'Note: This approach works, it has O(nlongn) time complexity and beated other submissions by 89% in Runtine and 22% in Memory'


#     'Min-heap approach'
#     def kthSmallest(matrix: list[list[int]], k: int) -> int:

#         # Capture the matrix dimentions
#         n = len(matrix)

#         # Import the heapq module
#         import heapq
        
#         # Create a min-heap with the first element of each row
#         min_heap = [(matrix[i][0], i, 0) for i in range(n)]
#         heapq.heapify(min_heap)
        
#         # Extract min k-1 times to get to the kth smallest element
#         for _ in range(k - 1):
#             value, row, col = heapq.heappop(min_heap)
#             if col + 1 < n:
#                 heapq.heappush(min_heap, (matrix[row][col + 1], row, col + 1))
        
#         # The root of the heap is the kth smallest element
#         return heapq.heappop(min_heap)[0]

#     # Testing
#     print(kthSmallest(matrix=matrix, k=k))

#     'Note: This solution worked, it has a time complexity of O(klogn) and beated submissions by 50% in Runtime and 34% in Memory.'

#     'Done'




'''32. Longest Valid Parentheses'''
# def x():

#     from typing import Optional

#     # Input
#     # # Case 1
#     # s = "(()"
#     # # Output: 2

#     # Case 2
#     s = ")()())"
#     # Output: 4
    
#     # # Case 3
#     # s = ""
#     # # Output: 0
           
#     # # Custom Case
#     # s = "()"
#     # # Output: 0

#     '''
#     My Approach (Stack)

#         Intuition:
            
#             Based on what I learnt in the '20. Valid Parentheses' past leetcode challenge, I will try to modify it 
#             to make it work for this use case.
#     '''

#     def longestValidParentheses(s: str) -> int:

#         # Define the max string length holder
#         max_len = 0

#         # Handle Corner case: Empty string
#         if not s:
#             return max_len
        
#         # Initialize the variables to work with
#         stack = list(s)     # Generate a stack with the full input
#         temp = []           # Create a temp holder to check parentheses validity
#         temp_count = 0      # Create a temporary count to keep record of the running longest valid parentheses before uptading max_len

#         # Go thru the string character by character from right to left
#         while stack:

#             popped = stack.pop(-1)

#             # If the last popped char is a closing one, store it in the temp holder
#             if stack and popped == ')':
#                 temp.insert(0,popped)
            
#             else:
#                 # If the last stored char doens't match with the recently popped, means not a valid parentheses and the subtring to the right won't count for the next valid parenthesis, so update the running count and max count and reset the running count
#                 if not stack or not temp or popped == ')':
#                     max_len = max(max_len, temp_count)  # Update the max count to hold the max between the current max and the running max
#                     temp_count = 0                      # Reset the running count to start fresh a new count for the remaining string
#                     temp.clear()                        # Clear up the temp holder to star anew the running count            
                
#                 # Otherwise is a valid match
#                 else:
#                     temp_count += 2     # Add 2 to the temporary count, since '()' counts for 2
#                     temp = temp[1:]     # Take out the valid closing char from the temp holder   

#         # Return 'max_len'
#         return max_len

#     # Testing
#     print(longestValidParentheses(s=s))

#     '''
#     Note:     
#         This approach only solve 54% of test cases and has some issues managing the stack, opening parentheses and the frequency of the temp holders resetting.
#         Apparently there is a better way to manage this by tracking the string indices based on the same stack idea.
                
#     '''



#     '''
#     Optimized Approach (Stack)

#         Explanation of the new approach:

#             1. Using a stack of indices: We store the index of the last unmatched closing parenthesis ')' or the index of an unmatched opening parenthesis '('. This helps us calculate the length of valid substrings efficiently.

#             2. Initializing with -1: This handles the edge case where the string starts with a valid sequence. By initializing the stack with -1, we can easily calculate the length of the first valid substring.

#             3. Tracking indices, not characters: Instead of managing the characters directly, we work with the indices of parentheses. 
#                 This allows us to calculate the lengths of valid substrings by subtracting the index of the last unmatched parenthesis from the current index.

#             4. Calculating valid substring lengths: Each time a valid pair of parentheses is found (when the stack is not empty after popping), the length of the valid substring is i - stack[-1], 
#                 where i is the current index and stack[-1] is the index of the last unmatched parenthesis.

#             5. Edge cases: When encountering an unmatched closing parenthesis, we push its index onto the stack to handle any future valid subsequences that might occur after it.
#     '''

#     def longestValidParentheses(s: str) -> int:

#         # Initialize a stack with -1 to handle edge cases
#         stack = [-1]
#         max_len = 0

#         # Traverse the string
#         for i, char in enumerate(s):

#             if char == '(':
#                 # Push the index of the opening parenthesis
#                 stack.append(i)

#             else:
#                 # Pop the stack for a closing parenthesis
#                 stack.pop()

#                 if not stack:
#                     # If the stack is empty, push the current index
#                     stack.append(i)

#                 else:
#                     # Calculate the length of the valid substring
#                     max_len = max(max_len, i - stack[-1])
        
#         return max_len

#     # Testing
#     print(longestValidParentheses(s=s))

'''394. Decode String'''
# def x():
    
#     from typing import Optional

#     # Input
#     # Case 1
#     s = "3[a]2[bc]"
#     # Output: "aaabcbc"
    
#     # Case 2
#     s = "3[a2[c]]"
#     # Output: "accaccacc"
    
#     # Case 3
#     s = "2[abc]3[cd]ef"
#     # Output: "abcabccdcdcdef"

#     '''
#     Recursive Solution

#         Explanation:

#             - Digit Processing:
#                 * When we encounter a digit, we calculate the full number, which is k.
#                 * Then, i is incremented to skip the digit characters.
            
#             - Recursive Call for Nested Strings:
#                 * After encountering [, we call the helper function recursively to decode the substring within the brackets.
#                 * The recursive call returns the decoded substring and the updated position i after the closing bracket ].
            
#             - Concatenation:
#                 * The decoded string is then repeated k times and concatenated to the result.
            
#             - Bracket Management:
#                 * If we encounter a ], it means we've completed processing this section, so we return the result and the updated position i.
#     '''

#     def decodeString(s: str) -> str:

#         # Helper function with an index tracker
#         def helper(i):

#             result = ""

#             while i < len(s):

#                 if s[i].isdigit():

#                     # Extract the number (can be more than one digit)
#                     k = 0
#                     while s[i].isdigit():
#                         k = k * 10 + int(s[i])
#                         i += 1

#                     # Skip the '[' character
#                     i += 1
                    
#                     # Decode the substring within the brackets by calling helper recursively
#                     decoded_string, i = helper(i)
                    
#                     # Repeat and add to the result
#                     result += k * decoded_string
                    
#                 elif s[i] == ']':

#                     # End of this recursive call
#                     return result, i + 1
                
#                 else:
                    
#                     # Regular characters, just add them to the result
#                     result += s[i]
#                     i += 1
            
#             return result, i
    
#         # Start the recursive decoding
#         decoded_string, _ = helper(0)

#         return decoded_string

#     # Testing
#     # print(decodeString(s=s))

    
    
    
#     '''
#     Iterative Solution (Stack)

#         Explanation:

#             - Digit Processing:
#                 * Similar to the recursive approach, this part extracts numbers. If the number is multi-digit, it handles that correctly.
#                 * The number is pushed onto count_stack.
            
#             - Handling [:
#                 * When encountering [, push the current result onto result_stack and reset result to build a new substring for this nested level.
            
#             - Handling ]:
#                 * When encountering ], pop from count_stack to get how many times to repeat result, and from result_stack to get the previous string context.
#                 * Update result by appending the repeated substring to the last saved result.
            
#             - Character Handling:
#                 * Characters are added to result as they are encountered, building up the decoded string layer by layer.
#     '''

#     def decodeString(s: str) -> str:

#         # Stacks for numbers and results
#         count_stack = []
#         result_stack = []
#         result = ""
#         i = 0

#         while i < len(s):

#             if s[i].isdigit():

#                 # Calculate the full number, which may be more than one digit
#                 k = 0
#                 while i < len(s) and s[i].isdigit():
#                     k = k * 10 + int(s[i])
#                     i += 1

#                 count_stack.append(k)
            
#             elif s[i] == '[':
                
#                 # Push the current result to the stack and reset it for new nested level
#                 result_stack.append(result)
#                 result = ""
#                 i += 1

#             elif s[i] == ']':
                
#                 # Pop the last count and repeat the current result that many times
#                 repeat_times = count_stack.pop()
#                 last_result = result_stack.pop()
#                 result = last_result + result * repeat_times
#                 i += 1

#             else:
                
#                 # Regular characters, just add to the result
#                 result += s[i]
#                 i += 1

#         return result

#     # Testing
#     print(decodeString(s=s))

'''739. Daily Temperatures'''
# def x():
    
#     from typing import Optional

#     # Input
#     # Case 1
#     temperatures = [73,74,75,71,69,72,76,73]
#     # Output: [1,1,4,2,1,1,0,0]

#     # # Case 2
#     # temperatures = [30,40,50,60]
#     # # Output: [1,1,1,0]

#     # # Case 3
#     # temperatures = [30,60,90]
#     # # Output: [1,1,0]


#     '''
#     Solution (Monotonic Stack)

#         Explanation:
            
#             1. Initialize a Stack: Start by creating an empty stack that will hold the indices of days with temperatures that haven't yet found a warmer day.

#             2. Traverse from Left to Right: For each day's temperature, check if it's higher than the temperature at the index stored on top of the stack.

#                 - If it is, it means we've found a "warmer day" for all indices in the stack with lower temperatures.
#                 - Pop from the stack, calculate the difference in indices (i.e., days until a warmer temperature), and store it in the result array.
            
#             3. Store the Result: If the current day's temperature is not warmer than the temperature at the top index in the stack, push the current day's index onto the stack and continue.

#             4. End of Loop: By the end of the loop, any remaining indices in the stack represent days that don't have a warmer day after them, so they'll remain 0 in the result array (default value).
#     '''

#     def dailyTemperatures(temperatures: list[int]) -> list[int]:
        
#         # Capture the length of the input
#         n = len(temperatures)

#         # Initialize the stacks
#         result = [0]*n
#         stack = []

#         # Process the input
#         for i in range(n):

#             while stack and temperatures[i] > temperatures[stack[-1]]:

#                 prev_day = stack.pop()
#                 result[prev_day] = i - prev_day

#             stack.append(i)
                
#         # Return the processed result holder
#         return result

#     # Testing
#     print(dailyTemperatures(temperatures=temperatures))

#     '''Note: Done'''

'''71. Simplify Path'''
# def x():
    
#     from typing import Optional

#     # Input
#     # Case 1
#     path = "/home/"
#     # Output: "/home"

#     # Case 2
#     path = "/home//foo/"
#     # Output: "/home/foo"

#     # Case 3
#     path = "/home/user/Documents/../Pictures"
#     # Output: "/home/user/Pictures"

#     # Case 4
#     path = "/../"
#     # Output: "/"

#     # Case 5
#     path = "/.../a/../b/c/../d/./"
#     # Output: "/.../b/d"

#     '''
#     My Approach

#         1. Split the path by / to process each component.
#         2. Traverse the components:
#             + Push valid directory names onto the stack.
#             + Pop the stack for .. (if the stack isn't empty).
#             + Ignore '.' or empty components.
#         3. Join the stack with / to construct the simplified path.
#     '''

#     def simplifyPath(path: str) -> str:

#         # Redefine the input path
#         components = path.split('/')

#         # Initialize a 'stack' list holder
#         stack = []

#         # Process the input
#         for comp in components:

#             if comp == '..':
#                 if stack:
#                     stack.pop()
            
#             elif comp and comp != '.':
#                 stack.append(comp)        
        
#         return '/'+ '/'.join(stack)


#     # Testing
#     print(simplifyPath(path=path))

#     '''Note: Done'''
















