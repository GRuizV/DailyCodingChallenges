'''
CHALLENGES INDEX

146. LRU Cache (Implementation)


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


(1)
'''


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None




'''146. LRU Cache'''
# def x():

#     from typing import Optional

#     # Input
#     # Case 1
#     commands = ["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
#     values = [[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
#     # Output: [null, null, null, 1, null, -1, null, -1, 3, 4]

#     # Case 15
#     commands = ["LRUCache","get","put","get","put","put","get","get"]
#     values = [[2],[2],[2,6],[1],[1,5],[1,2],[1],[2]]
#     # Output: [null,-1,null,-1,null,null,2,6]


#     '''
#     My Approach
    
#         Intuition:

#             - Use the 'insert' built-in list method to make two lists 'keys' and 'values' act as a queue to keep track of the order
#                 of the inserted values and to simulate the dict queries to make O(1) time when using get method.
#     '''

#     class LRUCache(object):   

#         def __init__(self, capacity: int):
#             self.capacity = capacity
#             self.keys = []
#             self.values = []
        

#         def get(self, key: int) -> int:

#             if key not in self.keys:
#                 return -1
            
#             else:
                
#                 # Get the current index for the entry
#                 idx = self.keys.index(key)

#                 # Get the value for the key passed
#                 value = self.values[idx]

#                 # Update the lists according to this query
#                 self.keys = [self.keys[idx]] + self.keys[:idx] + self.keys[idx+1:]
#                 self.values = [self.values[idx]] + self.values[:idx] + self.values[idx+1:]
            
#             # Return the value found
#             return value


#         def put(self, key: int, value: int) -> None:
            
#             # If key's already in the list
#             if key in self.keys:

#                 # Get the current index for the entry
#                 idx = self.keys.index(key)

#                 # Update the lists according to this query
#                 self.keys = [self.keys[idx]] + self.keys[:idx] + self.keys[idx+1:]
#                 self.values = [value] + self.values[:idx] + self.values[idx+1:]

#                 # Finish the operation
#                 return

#             # If capacity is at limit
#             if len(self.values) == self.capacity:

#                 # Take the least recent entries out
#                 self.keys = self.keys[:-1]
#                 self.values = self.values[:-1]

#             # Update the Cache with the input entry
#             self.keys.insert(0, key)
#             self.values.insert(0, value)

#     # Testing
#     res = []
#     for i,com in enumerate(commands):

#         if com == 'LRUCache':
#             obj = LRUCache(values[i][0])
#             res.append(None)
        
#         elif com == 'put':
#             obj.put(values[i][0], values[i][1])
#             res.append(None)
        
#         else:
#             res.append(obj.get(values[i][0]))


#     print(res)

#     '''Note: There a more efficient way to solve this challenge using Doubly Linked List, but I am seeing it as a stretch for this kind of challenges'''

















