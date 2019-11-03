import java.util.ArrayList;
import java.util.List;


public class DecisionTreeImpl {
	public DecTreeNode root;
	public List<List<Integer>> trainData;
	public int maxPerLeaf;
	public int maxDepth;
	public int numAttr;

	// Build a decision tree given a training set
	DecisionTreeImpl(List<List<Integer>> trainDataSet, int mPerLeaf, int mDepth) {
		this.trainData = trainDataSet;
		this.maxPerLeaf = mPerLeaf;
		this.maxDepth = mDepth;
		if (this.trainData.size() > 0) this.numAttr = trainDataSet.get(0).size() - 1;
		this.root = buildTree(this.trainData, 0);
	}
	
	private DecTreeNode buildTree(List<List<Integer>> trainData, int depth) {
		
		if(depth >= this.maxDepth || trainData.size() < this.maxPerLeaf) {
			int count1 = 0;
			for(int i = 0; i < trainData.size(); i++) {
				if(trainData.get(i).get(trainData.get(0).size() - 1) == 1) count1++;
			}
			int label = 0;
			if(count1 >= ((double) trainData.size() / 2.0) ) label = 1;
			return new DecTreeNode(label, 0, 0);
		}
		if(trainData.size() == 1) {
			return new DecTreeNode(trainData.get(0).get(trainData.get(0).size()-1), 0, 0);
		}
		
		//Find which atribute and threshold have highest information gain
		Double minEntropy = 9999.0;
		int attribute = 0;
		int threshhold = 0;
		Double count = (double) trainData.size();
		Double countLess = 0.0;
		Double countGreater = 0.0;
		Double countL1 = 0.0;
		Double countL0 = 0.0;
		Double countG0 = 0.0;
		Double countG1 = 0.0;
		Double entropy = 0.0;
		for(int j = 0; j < trainData.get(0).size() - 1; j++) {
			for(int k = 1; k < 10; k++) {
				countLess = 0.0;
				countGreater = 0.0;
				countL1 = 0.0;
				countL0 = 0.0;
				countG0 = 0.0;
				countG1 = 0.0;
				for(int i = 0; i < trainData.size(); i++) {
					if(trainData.get(i).get(j) <= k) {
						countLess++;
						if(trainData.get(i).get(trainData.get(0).size() - 1) == 0) countL0++;
						else countL1++;
					}
					else {
						countGreater++;
						if(trainData.get(i).get(trainData.get(0).size() - 1) == 0) countG0++;
						else countG1++;
					}
				}
				
				entropy = -1 *( (countLess/count)*((countL0/countLess)*(log2(countL0/countLess)) + 
						(countL1/countLess)*(log2(countL1/countLess)) )+
						(countGreater/count)*((countG0/countGreater)*(log2(countG0/countGreater)) + 
						(countG1/countGreater)*(log2(countG1/countGreater))));

				if(entropy < minEntropy) {
					minEntropy = entropy;
					attribute = j;
					threshhold = k;
				}
			}
		}
		
		Double prevEntropy = -1*((countL0+countG0)/count)*log2((countL0+countG0)/count) - 
				((countL1 + countG1 )/ count)*log2(((countL1 + countG1 )/ count));

		if(minEntropy >= prevEntropy) {
			int count1 = 0;
			for(int i = 0; i < trainData.size(); i++) {
				if(trainData.get(i).get(trainData.get(0).size() - 1) == 1) count1++;
			}
			int label = 0;
			if(count1 >= ((double) trainData.size() / 2.0) ) label = 1;
			return new DecTreeNode(label, 0, 0);
		}
		
		DecTreeNode node = new DecTreeNode(this.trainData.get(0).get(0), attribute, threshhold);
		List<List<Integer>> lessData = new ArrayList<List<Integer>>();
		List<List<Integer>> greaterData = new ArrayList<List<Integer>>();
		for(int i = 0; i < trainData.size(); i++) {
			if(trainData.get(i).get(attribute) <= threshhold) {
				lessData.add(trainData.get(i));
			}
			else {
				greaterData.add(trainData.get(i));
			}
		}
		node.left = buildTree(lessData, depth+1);
		node.right = buildTree(greaterData, depth+1);
		
		return node;
	}
	
	public Double log2(Double x) {
		if (x ==0.0) return 0.0;
		return (Math.log(x)/Math.log(2));
	}
	
	public int classify(List<Integer> instance) {
		DecTreeNode node = this.root;
		while(node.isLeaf() != true) {
			if(instance.get(node.attribute) <= node.threshold) node = node.left;
			else node = node.right;
		}
		return(node.classLabel);
	}
	
	// Print the decision tree
	public void printTree() {
		printTreeNode("", this.root);
	}

	public void printTreeNode(String prefixStr, DecTreeNode node) {
		String printStr = prefixStr + "X_" + node.attribute;
		System.out.print(printStr + " <= " + String.format("%d", node.threshold));
		if(node.left.isLeaf()) {
			System.out.println(" : " + String.valueOf(node.left.classLabel));
		}
		else {
			System.out.println();
			printTreeNode(prefixStr + "|\t", node.left);
		}
		System.out.print(printStr + " > " + String.format("%d", node.threshold));
		if(node.right.isLeaf()) {
			System.out.println(" : " + String.valueOf(node.right.classLabel));
		}
		else {
			System.out.println();
			printTreeNode(prefixStr + "|\t", node.right);
		}
	}
	
	public double printTest(List<List<Integer>> testDataSet) {
		int numEqual = 0;
		int numTotal = 0;
		for (int i = 0; i < testDataSet.size(); i ++)
		{
			int prediction = classify(testDataSet.get(i));
			int groundTruth = testDataSet.get(i).get(testDataSet.get(i).size() - 1);
			System.out.println(prediction);
			if (groundTruth == prediction) {
				numEqual++;
			}
			numTotal++;
		}
		double accuracy = numEqual*100.0 / (double)numTotal;
		System.out.println(String.format("%.2f", accuracy) + "%");
		return accuracy;
	}
}
