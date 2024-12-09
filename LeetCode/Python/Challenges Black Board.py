'''
CHALLENGES TAGS

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

'''

#Template
'xxx'
'''xxx'''
def x():
    
    from typing import Optional

    # Input
    # Case 1
    head = None
    # Output: None

    '''
    My Approach

        Intuition:
            
            -...
    '''

    def x() -> int:

        # Handle Corner case: ...
        if not head:
            return
                
        # Return ...
        return 

    # Testing
    print(x())

    '''Note: Done'''


# In case is a Tree Challenge
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def pretty_print_bst(node:TreeNode, prefix="", is_left=True):

    if not node:
        return

    if node.right is not None:
        pretty_print_bst(node.right, prefix + ("│   " if is_left else "    "), False)

    print(prefix + ("└── " if is_left else "┌── ") + str(node.val))

    if node.left is not None:
        pretty_print_bst(node.left, prefix + ("    " if is_left else "│   "), True)



'55. Jump Game (Array) (DP) (GRE)'
'''55. Jump Game'''
def x():
    
    from typing import Optional

    # Input
    # Case 1
    nums = [2,3,1,1,4]
    # Output: True

    # Case 2
    nums = [3,2,1,0,4]
    # Output: False

    '''
    My Approach (DFS)

        Intuition:

            - Serialize 'nums' with enumerate in a 's_nums' holder.
            - Initialize a list 'stack' initialized in the first element of 's_nums'         
            - In a while loop ('stack exists'):
                + Initialize a position holder 'curr' with stack.pop(0).
                + if 'curr' holder first element (item position) added to its second element (item value) reaches the end of the list return True
                    if ['curr'[0] + 'curr'[1]] >= len(nums):
                        * Return True
                + Extend the stack to all reachable positions ( stack.extend([s_nums[curr[0]+x] for x in range(1, curr[1]+1)]) if curr[0]+x <= len(nums))

            - If the code gets out of the loop, means the end of the list is out of reach, return False then.
    '''

    def canJump(nums: list[int]) -> bool:

        # Serialize 'nums' in 's_nums'
        s_nums = [(i,v) for i,v in enumerate(nums)]

        # Initialize an stack list at the first s_num value
        stack = [s_nums[0]]

        while stack:
            
            curr = stack.pop(0)

            if curr[0]+curr[1] >= len(nums)-1:
                return True

            stack.extend([s_nums[curr[0]+x] for x in range(1, curr[1]+1) if curr[0]+x <= len(nums) and s_nums[curr[0]+x] not in stack] )

        # Return False if the code gets up to here
        return False

    # Testing
    print(canJump(nums = nums))

    '''Note: Done'''








x()






















































