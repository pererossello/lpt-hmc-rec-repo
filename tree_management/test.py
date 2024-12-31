import os
import sys

from main import HMCTree

GROUND_DIR = 'tree_management/EXP_A'
Tree = HMCTree(GROUND_DIR)

# Tree.print_tree()
# Tree.save_tree_to_txt()

Tree.create_n_roots(2)

# Tree.create_n_branches_at_tips(1)

# print()