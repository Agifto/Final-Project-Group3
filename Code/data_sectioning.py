# Import packages
import random
import os
import shutil
from fractions import Fraction

# Function that moves random files from one directory to another
def move_random_files(path_from, path_to, n):
    files = os.listdir(path_from)
    files.sort()

    for to_move in random.sample(files, int(len(files) * n)):
        f = to_move
        src = path_from + f
        dst = path_to
        shutil.move(src, dst)

# Separate out 30% of our data from Training set for Validation & Testing set
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Train/Chaetognaths/', path_to='/home/ubuntu/Deep-Learning/final_project/Test/Chaetognaths/', n=.3)
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Train/Crustaceans/', path_to='/home/ubuntu/Deep-Learning/final_project/Test/Crustaceans/', n=.3)
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Train/Detritus/', path_to='/home/ubuntu/Deep-Learning/final_project/Test/Detritus/', n=.3)
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Train/Diatoms/', path_to='/home/ubuntu/Deep-Learning/final_project/Test/Diatoms/', n=.3)
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Train/Gelatinous_Zooplankton/', path_to='/home/ubuntu/Deep-Learning/final_project/Test/Gelatinous_Zooplankton/', n=.3)
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Train/Other_Invert_Larvae/', path_to='/home/ubuntu/Deep-Learning/final_project/Test/Other_Invert_Larvae/', n=.3)
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Train/Protists/', path_to='/home/ubuntu/Deep-Learning/final_project/Test/Protists/', n=.3)
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Train/Trichodesmium/', path_to='/home/ubuntu/Deep-Learning/final_project/Test/Trichodesmium/', n=.3)

# From the separated 30%, take out 2/3rds of it for Validation set
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Test/Chaetognaths/', path_to='/home/ubuntu/Deep-Learning/final_project/Validation/Chaetognaths/', n=Fraction(2,3))
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Test/Crustaceans/', path_to='/home/ubuntu/Deep-Learning/final_project/Validation/Crustaceans/', n=Fraction(2,3))
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Test/Detritus/', path_to='/home/ubuntu/Deep-Learning/final_project/Validation/Detritus/', n=Fraction(2,3))
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Test/Diatoms/', path_to='/home/ubuntu/Deep-Learning/final_project/Validation/Diatoms/', n=Fraction(2,3))
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Test/Gelatinous_Zooplankton/', path_to='/home/ubuntu/Deep-Learning/final_project/Validation/Gelatinous_Zooplankton/', n=Fraction(2,3))
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Test/Other_Invert_Larvae/', path_to='/home/ubuntu/Deep-Learning/final_project/Validation/Other_Invert_Larvae/', n=Fraction(2,3))
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Test/Protists/', path_to='/home/ubuntu/Deep-Learning/final_project/Validation/Protists/', n=Fraction(2,3))
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Test/Trichodesmium/', path_to='/home/ubuntu/Deep-Learning/final_project/Validation/Trichodesmium/', n=Fraction(2,3))

# We thus end up with 70% in our Training set, 20% in our Validation set, and 10% in our Testing set