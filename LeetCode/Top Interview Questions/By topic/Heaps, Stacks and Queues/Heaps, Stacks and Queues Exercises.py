'''
CHALLENGES INDEX

20. Valid Parentheses (Stack)
23. Merge k Sorted Lists (LL) (DQ) (Heap) (Sorting)
42. Trapping Rain Water (Array) (TP) (DS) (Stack)
150. Evaluate Reverse Polish Notation (Stack)
155. Min Stack (Stack)


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

(XX)
'''



'20. Valid Parentheses'
# def x():

#     # input / Case - expected result
#     s = '()'    # True
#     s = '()[]{}'    # True
#     s = '(]'    # False
#     s = '([({[]{}}())])'    # True
#     s = '([({[)]{}}())])'    # False
#     s = '))'    # False
#     s = '(('    # False

#     # My approach
#     def isValid(s):

#         stack = list(s)
#         temp = []
#         dic = {'(': ')', '[':']', '{':'}'}  

#         while True:

#             if len(stack) == 0 and len(temp) != 0:
#                 return False

#             popped = stack.pop(-1)

#             if popped in '([{':
                
#                 if len(temp) == 0 or temp[0] != dic[popped]:
#                     return False
                                
#                 else:                
#                     temp = temp[1:]

#             else:
#                 temp.insert(0,popped)

#             if len(stack) == 0 and len(temp)==0:
#                 return True  

#     # Testing
#     print(isValid(s))

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


















