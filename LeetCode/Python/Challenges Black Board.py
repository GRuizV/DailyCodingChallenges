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




'205. Isomorphic Strings (Hash Table)'
'''205. Isomorphic Strings'''
def x():
    
    from typing import Optional

    # Input
    # Case 1
    s = "egg"
    t = "add"
    # Output: True

    # # Case 2
    # s = "foo"
    # t = "bar"
    # # Output: False

    # # Case 3
    # s = "paper"
    # t = "title"
    # # Output: True

    '''
    My Approach
    '''

    def isIsomorphic(s: str, t: str) -> bool:

        # Capture the number of characters in each string
        s_set, t_set = set(s), set(t)

        # Handle Corner case: different lengths between sets
        if len(s_set) != len(t_set):
            return False

        # Build the dictionary between them
        dic = {k:v for k,v in zip(s, t)}

        # Initialize a string result holder to build the resulting string to compare
        result = ''
        
        # Build the resulting string based on the characters of 's' translated with the 'dic'
        for char in s:
            result += dic[char]

        # Return the comparison of the result string and 't'
        return result == t


    # Testing
    print(isIsomorphic(s=s, t=t))

    '''Note: Done'''





x()






















































