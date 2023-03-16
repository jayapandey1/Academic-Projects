from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat, ImageDraw
import numpy as np
import time
import cv2

class Agent:
    
    def __init__(self):
        pass

    def Solve(self, problem):
        if problem.problemSetName == "Basic Problems B" or problem.problemSetName == "Test Problems B" or problem.problemSetName == "Challenge Problems B" or problem.problemSetName == "Raven's Problems B":   
            return self.category1_Problems(problem)
        elif problem.problemSetName == "Basic Problems C" or problem.problemSetName == "Test Problems C" or problem.problemSetName == "Challenge Problems C" or problem.problemSetName == "Raven's Problems C":
            answer = self.category2_Problems(problem)
            return answer
        else:
            return self.category3_Problems(problem)   

    def check_ifSame(self, imageA, imageB, range): 
        if self.ratio_whiteandBlack(ImageChops.difference(imageA, imageB)) < range:
            return True
        else:
            return False

    def ratio_whiteandBlack(self, image): 
        w = 1
        b = 1
        for value in image.getdata():
            if value > 0:  
                w += 1
            else:
                b += 1
        return w / b
    
    def diff_darkPixelRatio(self, ref1, ref2):
        return (np.sum(ref1)/np.size(ref1)) - (np.sum(ref2)/np.size(ref2))

    def diff_IntersectionRatio(self, ref1, ref2):
        return (np.sum(cv2.bitwise_or(ref1, ref2))/np.sum(ref1)) - (np.sum(cv2.bitwise_or(ref1, ref2))/np.sum(ref2))
    
    def Binarize(self, image):
        val, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
        return image

    def darken(self, image):
        width, height = image.size
        ImageDraw.floodfill(image, xy = (int(0.5 * width), int(0.5 * height)), value = 0) 
        return image
    
    def darkPixelFill(self, ImageA, ImageB, ImageC, options): 
        imageFill = []
        if self.ratio_whiteandBlack(ImageA) > self.ratio_whiteandBlack(ImageB):
            if self.check_ifSame(self.darken(ImageA), ImageB, 0.045):
                for option in options:
                    if self.check_ifSame(self.darken(ImageC), option, 0.045):
                        imageFill.append(5)
                    else:
                        imageFill.append(0)
            else:
                imageFill = [0, 0, 0, 0, 0, 0]
        else:
            imageFill = [0, 0, 0, 0, 0, 0]

        return imageFill

    def option_weightage(self, ImageA,ImageB,ImageC, opList): 

        rotation_weightage = np.array(self.rotation_weight(ImageA,ImageB,ImageC, opList)) + np.array(self.rotation_weight(ImageA,ImageC,ImageB, opList))
        reflection_weightage = np.array(self.reflection_weight(ImageA,ImageB,ImageC, opList)) + np.array(self.reflection_weight(ImageA,ImageC,ImageB, opList))
        difference_weightage = np.array(self.compare_difference(ImageA,ImageB,ImageC, opList)) + np.array(self.compare_difference(ImageA,ImageC,ImageB, opList))
        darkPixel_weightage = np.array(self.darkPixelFill(ImageA,ImageB,ImageC, opList)) + np.array(self.darkPixelFill(ImageA,ImageC,ImageB, opList))

        eachOptionWeightage = list(2*reflection_weightage + rotation_weightage + difference_weightage + darkPixel_weightage)
        return eachOptionWeightage 

    def option_weightage_category3(self, ImageA, ImageB, ImageC, ImageD, ImageE, ImageF, ImageG, ImageH, options, problem):  
        
        rotation_weightage = np.array(self.rotation_weight_category3(ImageA, ImageC, ImageG, options))
        difference_weightage = np.array(self.diffcheck(ImageC, ImageF, options)) + np.array(self.diffcheck(ImageG, ImageH, options)) + \
            np.array(self.diffcheck_diff(ImageA, ImageB, ImageC, ImageG, ImageH, options)) +np.array(self.diffcheck(ImageA, ImageE, options))
        reflection_weightage = np.array(self.reflection_score_category3(ImageA, ImageB, ImageC, ImageG, options))
        pixel_comparison_row_weightage = np.array(self.pixel_comparison(ImageA, ImageB, ImageC, ImageD, ImageE, ImageF, ImageG, ImageH, options))
        pixel_comparison_col_weightage = np.array(self.col_pixel_compare(ImageA, ImageB, ImageC, ImageD, ImageE, ImageF, ImageG, ImageH, options))
        diagonal_weightage = np.array(self.diagonal_pixel_compare(ImageA, ImageB, ImageC, ImageD, ImageE, ImageF, ImageG, ImageH, options))
    
        if problem.problemSetName == "Basic Problems E" or problem.problemSetName == "Test Problems E" or problem.problemSetName == "Challenge Problems E" or problem.problemSetName == "Raven's Problems E":
            eachOptionWeightage = list( rotation_weightage + 2 * difference_weightage +pixel_comparison_row_weightage + reflection_weightage )
        else:
            eachOptionWeightage = list( rotation_weightage + 2 * difference_weightage +pixel_comparison_row_weightage + reflection_weightage + pixel_comparison_col_weightage  )
        
        return eachOptionWeightage

    def category1_Problems(self, problem):
        options = []
        for i in range(1, 7):
            opt = Image.open(problem.figures.get(str(i)).visualFilename).convert('L')
            options.append(opt)
        ImageA = Image.open(problem.figures['A'].visualFilename).convert('L')
        ImageB = Image.open(problem.figures['B'].visualFilename).convert('L')
        ImageC = Image.open(problem.figures['C'].visualFilename).convert('L')

        weightage = self.option_weightage(ImageA, ImageB, ImageC, options)
        maxScore = max(weightage)
        answer = weightage.index(maxScore) + 1
        return answer 

    def category2_Problems(self, problem):
        options = []
        darkRatio_optionValues = []
        intersection_optionValues = []
        optionValues_inRange =[]
        for i in range(1,9):
            opt = self.Binarize(cv2.imread(problem.figures.get(str(i)).visualFilename, 0))
            options.append(opt)
        ImageG = self.Binarize(cv2.imread(problem.figures['G'].visualFilename, 0))
        ImageH = self.Binarize(cv2.imread(problem.figures['H'].visualFilename, 0))

        darkRatio_GHValues = self.diff_darkPixelRatio(ImageG, ImageH)
        intersection_GHValues = self.diff_IntersectionRatio(ImageG, ImageH)

        for opt in options:
            darkRatio_optionValues.append(self.diff_darkPixelRatio(ImageH, opt))
            intersection_optionValues.append(self.diff_IntersectionRatio(ImageH, opt))

        for index, optionval in enumerate(darkRatio_optionValues):
            if darkRatio_GHValues - 2 <= optionval <= darkRatio_GHValues + 2:
                optionValues_inRange.append(intersection_optionValues[index])

        if len(optionValues_inRange) > 0:
            index, value = min(enumerate(optionValues_inRange),key=lambda x: abs(x[1]-intersection_GHValues))
            index = intersection_optionValues.index(value)
            return index+1
        else:
            index = np.argmin(np.abs(darkRatio_optionValues-darkRatio_GHValues))
            return index + 1

    def category3_Problems(self, problem):
        #start = time.time()
        options = []
        for i in range(1,9):
            opt = Image.open(problem.figures.get(str(i)).visualFilename).convert('L')
            options.append(opt)
        ImageA = Image.open(problem.figures['A'].visualFilename).convert('L')
        ImageB = Image.open(problem.figures['B'].visualFilename).convert('L')
        ImageC = Image.open(problem.figures['C'].visualFilename).convert('L')
        ImageD = Image.open(problem.figures['D'].visualFilename).convert('L')
        ImageE = Image.open(problem.figures['E'].visualFilename).convert('L')
        ImageF = Image.open(problem.figures['F'].visualFilename).convert('L')
        ImageG = Image.open(problem.figures['G'].visualFilename).convert('L')
        ImageH = Image.open(problem.figures['H'].visualFilename).convert('L')
        
        weightage = self.option_weightage_category3(ImageA, ImageB, ImageC, ImageD, ImageE, ImageF, ImageG, ImageH, options, problem)
        answer = weightage.index(max(weightage)) + 1
        #end = time.time()
        #run_time = str((end - start) * 1000)
        #print("Running time:" + run_time + "ms")
        return answer

    def rotation_weight(self, ImageA, ImageB, ImageC, options):  
        rotation_weight_val = []

        if self.check_ifSame(ImageA.rotate(270), ImageB, 0.1):
            for option in options:
                if self.check_ifSame(ImageC.rotate(270), option, 0.05):
                    rotation_weight_val.append(3)
                elif self.check_ifSame(ImageC.rotate(270), option, 0.1):
                    rotation_weight_val.append(1)
                else:
                    rotation_weight_val.append(0)
        else:
            rotation_weight_val = [0, 0, 0, 0, 0, 0]
        return rotation_weight_val

    def rotation_weight_category3(self, ImageA, ImageC, ImageG, options): 
        rotation_weight_val = []

        if self.check_ifSame(ImageA.transpose(Image.FLIP_LEFT_RIGHT), ImageC, 0.02): 
            rotation_weight_val = [0, 0, 0, 0, 0, 0, 0, 0]
        elif self.check_ifSame(ImageA.rotate(270), ImageC, 0.02):
            for option in options:
                if self.check_ifSame(ImageG.rotate(270), option, 0.02):
                    rotation_weight_val.append(6)
                else:
                    rotation_weight_val.append(0)
        else:
            rotation_weight_val = [0, 0, 0, 0, 0, 0, 0, 0]
        return rotation_weight_val

    def diffcheck(self, Image1, Image2, opList): 
        difference = []
        if (self.ratio_whiteandBlack(Image2) - self.ratio_whiteandBlack(Image1)) > 0.25:  
            for option in opList:
                if (self.ratio_whiteandBlack(option) - self.ratio_whiteandBlack(Image2)) > 0.4:  
                    if self.check_ifSame(ImageChops.difference(Image2, option), ImageChops.difference(Image1, Image2), 0.04):
                        difference.append(12)
                    elif self.check_ifSame(ImageChops.difference(Image2, option), ImageChops.difference(Image1, Image2), 0.06):
                        difference.append(10)
                    elif self.check_ifSame(ImageChops.difference(Image2, option), ImageChops.difference(Image1, Image2), 0.08):
                        difference.append(8)
                    elif self.check_ifSame(ImageChops.difference(Image2, option), ImageChops.difference(Image1, Image2), 0.1):
                        difference.append(3)
                    elif self.check_ifSame(ImageChops.difference(Image2, option), ImageChops.difference(Image1, Image2), 0.2):
                        difference.append(2)
                    else:
                        difference.append(1)
                else:
                    difference.append(0)

        elif (self.ratio_whiteandBlack(Image1) - self.ratio_whiteandBlack(Image2)) > 0.25:  
            for option in opList:
                if (self.ratio_whiteandBlack(Image2) - self.ratio_whiteandBlack(option)) > 0.4: 
                    if self.check_ifSame(ImageChops.difference(Image2, option), ImageChops.difference(Image1, Image2), 0.04):
                        difference.append(12)
                    elif self.check_ifSame(ImageChops.difference(Image2, option), ImageChops.difference(Image1, Image2), 0.06):
                        difference.append(10)
                    elif self.check_ifSame(ImageChops.difference(Image2, option), ImageChops.difference(Image1, Image2), 0.08):
                        difference.append(8)
                    elif self.check_ifSame(ImageChops.difference(Image2, option), ImageChops.difference(Image1, Image2), 0.1):
                        difference.append(3)
                    elif self.check_ifSame(ImageChops.difference(Image2, option), ImageChops.difference(Image1, Image2), 0.2):
                        difference.append(2)
                    else:
                        difference.append(1)
                else:
                    difference.append(0)
        else:
            difference = [0, 0, 0, 0, 0, 0, 0, 0]
        return difference

    def diffcheck_diff(self, ImageA, ImageB, ImageC, ImageG, ImageH, options): 
        diff = []
        if self.check_ifSame(ImageChops.invert(ImageChops.difference(ImageA, ImageB)), ImageC, 0.04):  
            for option in options:
                if self.check_ifSame(ImageChops.invert(ImageChops.difference(ImageG, ImageH)), option, 0.06): 
                    diff.append(15)
                else:
                    diff.append(0)
        else:
            diff = [0, 0, 0, 0, 0, 0, 0, 0]
        return diff
    
    def pixel_values(self, image):
        pix_val = 0
        for val in image.getdata():
            if val == 0: 
                pix_val = pix_val+ 1
        return pix_val / len(list(image.getdata()))
     
    def pixel_comparison(self, ImageA, ImageB, ImageC, ImageD, ImageE, ImageF, ImageG, ImageH, options):  
        pScore = []

        if abs((self.pixel_values(ImageA) + self.pixel_values(ImageB) + self.pixel_values(ImageC)) - (self.pixel_values(ImageD) + self.pixel_values(ImageE) + self.pixel_values(ImageF))) < 0.002:
            for option in options:      
                if abs((self.pixel_values(ImageG) + self.pixel_values(ImageH) + self.pixel_values(option)) - (self.pixel_values(ImageA) + self.pixel_values(ImageB) + self.pixel_values(ImageC))) < 0.002 and abs((self.pixel_values(ImageG) + self.pixel_values(ImageH) + self.pixel_values(option)) - (self.pixel_values(ImageD) + self.pixel_values(ImageE) + self.pixel_values(ImageF))) < 0.002:  
                    pScore.append(10)
                else:
                    pScore.append(0)
        elif abs((self.pixel_values(ImageA) + self.pixel_values(ImageB) + self.pixel_values(ImageC)) - (self.pixel_values(ImageD) + self.pixel_values(ImageE) + self.pixel_values(ImageF))) < 0.01:
            for option in options:
                if abs(self.pixel_values(ImageG) + self.pixel_values(ImageH) + self.pixel_values(option) - (self.pixel_values(ImageA) + self.pixel_values(ImageB) + self.pixel_values(ImageC))) < 0.035 and abs(self.pixel_values(ImageG) + self.pixel_values(ImageH) + self.pixel_values(option) - (self.pixel_values(ImageD) + self.pixel_values(ImageE) + self.pixel_values(ImageF))) < 0.035:  
                    pScore.append(8)
                elif abs(self.pixel_values(ImageG) + self.pixel_values(ImageH) + self.pixel_values(option) - (self.pixel_values(ImageA) + self.pixel_values(ImageB) + self.pixel_values(ImageC))) < 0.05 and abs(self.pixel_values(ImageG) + self.pixel_values(ImageH) + self.pixel_values(option) - (self.pixel_values(ImageD) + self.pixel_values(ImageE) + self.pixel_values(ImageF))) < 0.05: 
                    pScore.append(4)
                else:
                    pScore.append(0)
        else:
            pScore = [0, 0, 0, 0, 0, 0, 0, 0]
        return pScore   

    def reflection_score_category3(self, ImageA,ImageB,ImageC, ImageG, options):  
        horizontal_reflection = []

        if self.check_ifSame(ImageA.transpose(Image.FLIP_LEFT_RIGHT), ImageB, 0.03) and (self.ratio_whiteandBlack(ImageC) - self.ratio_whiteandBlack(ImageA)) > 0.12:
            for option in options:
                if (self.ratio_whiteandBlack(option) - self.ratio_whiteandBlack(ImageG)) > 0.12:
                    horizontal_reflection.append(15)
                else:
                    horizontal_reflection.append(0)
        elif self.check_ifSame(ImageA.transpose(Image.FLIP_LEFT_RIGHT), ImageC, 0.05):
            for option in options:
                if self.check_ifSame(ImageG.transpose(Image.FLIP_LEFT_RIGHT), option, 0.05):
                    horizontal_reflection.append(1)
                else:
                    horizontal_reflection.append(0)
        else:
            horizontal_reflection = [0, 0, 0, 0, 0, 0, 0, 0]
        return horizontal_reflection   

    def reflection_weight(self, ImageA,ImageB,ImageC, options): 
        horizontal_reflection = []  
        vertical_reflection = [] 

        if self.check_ifSame(ImageA.transpose(Image.FLIP_LEFT_RIGHT),ImageB, 0.05):
            vertical_reflection = [0, 0, 0, 0, 0, 0]
            for option in options:
                if self.check_ifSame(ImageC.transpose(Image.FLIP_LEFT_RIGHT), option, 0.05):
                    horizontal_reflection.append(1)
                else:
                    horizontal_reflection.append(0)
        elif self.check_ifSame(ImageA.transpose(Image.FLIP_TOP_BOTTOM),ImageB, 0.18):
            horizontal_reflection = [0, 0, 0, 0, 0, 0]
            for option in options:
                if self.check_ifSame(ImageC.transpose(Image.FLIP_TOP_BOTTOM), option, 0.01):
                    vertical_reflection.append(2)
                elif self.check_ifSame(ImageC.transpose(Image.FLIP_TOP_BOTTOM), option, 0.18):
                    vertical_reflection.append(1)
                else:
                    vertical_reflection.append(0)
        else:
            horizontal_reflection = [0, 0, 0, 0, 0, 0]
            vertical_reflection = [0, 0, 0, 0, 0, 0]

        refScore = list(np.array(horizontal_reflection) + np.array(vertical_reflection))
        return refScore

    def compare_difference(self, ImageA,ImageB,ImageC, options): 
        difference = []
        for option in options:
            if self.check_ifSame(ImageChops.difference(ImageA,ImageB), ImageChops.difference(ImageC, option), 0.04):
                difference.append(1)
            else:
                difference.append(0)
        return difference 

    def col_pixel_compare(self, ImageA, ImageB, ImageC, ImageD, ImageE, ImageF, ImageG, ImageH, options): 
        col_totalpixel = []

        if abs((self.pixel_values(ImageA) + self.pixel_values(ImageD) + self.pixel_values(ImageG)) - (self.pixel_values(ImageB) + self.pixel_values(ImageE) + self.pixel_values(ImageH))) < 0.002:  
            for opt in options:
                if abs((self.pixel_values(ImageC) + self.pixel_values(ImageF)) + self.pixel_values(opt) - (self.pixel_values(ImageA) + self.pixel_values(ImageD) + self.pixel_values(ImageG))) < 0.002 and abs((self.pixel_values(ImageC) + self.pixel_values(ImageF)) + self.pixel_values(opt) - (self.pixel_values(ImageB) + self.pixel_values(ImageE) + self.pixel_values(ImageH))) < 0.002:  
                    col_totalpixel.append(10)
                else:
                    col_totalpixel.append(0)
        elif abs((self.pixel_values(ImageA) + self.pixel_values(ImageD) + self.pixel_values(ImageG)) - (self.pixel_values(ImageB) + self.pixel_values(ImageE) + self.pixel_values(ImageH))) < 0.012:
            for opt in options:
                if abs((self.pixel_values(ImageC) + self.pixel_values(ImageF)) + self.pixel_values(opt) - (self.pixel_values(ImageA) + self.pixel_values(ImageD) + self.pixel_values(ImageG))) < 0.04 and abs((self.pixel_values(ImageC) + self.pixel_values(ImageF)) + self.pixel_values(opt) - (self.pixel_values(ImageB) + self.pixel_values(ImageE) + self.pixel_values(ImageH))) < 0.04:
                    col_totalpixel.append(8)
                elif abs((self.pixel_values(ImageC) + self.pixel_values(ImageF)) + self.pixel_values(opt) - (self.pixel_values(ImageA) + self.pixel_values(ImageD) + self.pixel_values(ImageG))) < 0.05 and abs((self.pixel_values(ImageC) + self.pixel_values(ImageF)) + self.pixel_values(opt) - (self.pixel_values(ImageB) + self.pixel_values(ImageE) + self.pixel_values(ImageH))) < 0.05:
                    col_totalpixel.append(4)
                else:
                    col_totalpixel.append(0)
        else:
            col_totalpixel = [0, 0, 0, 0, 0, 0, 0, 0]
        return col_totalpixel

    def diagonal_pixel_compare(self, A, B, C, D, E, F, G, H, opList):  
        pixA = self.pixel_values(A) 
        pixE =  self.pixel_values(E) 
        dia_pixel = []

        for op in opList:
            if abs(pixA - self.pixel_values(op)) < 0.002 and abs(pixE - self.pixel_values(op)) < 0.002:  
                dia_pixel.append(10)
            else:
                dia_pixel.append(0)
        return dia_pixel

            


        
    
   