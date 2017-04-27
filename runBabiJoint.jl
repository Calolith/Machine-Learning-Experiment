include("parseBabi.jl")
include("modelFunctions.jl")

using Knet, AutoGrad, ParseBabi, ModelFunctions

#################### Configurations ####################
	
	batchSize = 32
	nHops = 3
	nEpochs = 60
	lrDecayStep = 15
	lrDecayAmount = 0.5
	lr = 0.01
	gradClip = 50
	d = 50

	validationSetRatio = 0.1
	timeEmbedding = true
	isBagOfWords = false
	weightAdjacent = true
	linearStart = true
	addNonLinearity = false
	addLinearLayer = false
	amountOfNoise = 0.0 #TODO
	
########################################################

function main()
	global lr
	baseDir = "data/tasksv11/en/"

	# vocab = Dictionary( String -> Integer )
	vocab = createVocabJoint(baseDir)
	testData = Dict()
	for taskID in 1:20
		testData[taskID] = parseBabiTask(vocab, baseDir, taskID; isTest = true)
	end
	# Story = Matrix( Word, Sentence, Story )
	# QStory = Matrix( Word, QuestionIdx )
	# Questions = Matrix( Info, QuestionIdx ) : Info = Vec( Answer, SentencesBefore, StoryIdx ) 
	story, qstory, questions = parseBabiTaskJoint(vocab, baseDir)
	qCount = size(questions, 2)
	allQs = randperm(qCount)
	trainQs = allQs[1:Int64(floor(qCount * 0.9))]
	validationQs = allQs[Int64(floor(qCount * 0.9)) + 1:end]

	memorySize = min(50, size(story, 2))
	vocabSize = length(vocab)
	enableSoftmax = !linearStart
	arrayType = eval(parse(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"))

	if weightAdjacent
		param = initweightsAdj(vocabSize, memorySize, nHops, d, timeEmbedding, addLinearLayer; atype = arrayType)
	else
		param = initweightsLyr(vocabSize, memorySize, d, timeEmbedding, addLinearLayer; atype = arrayType)
	end

	#print("Report @time: ")
	#lossPer, acc = @time report(param, questions, qstory, story, vocabSize, memorySize, enableSoftmax, trainQs, arrayType)
	print("ReportVal @time: ")
	lossPerVal, accVal = @time report(param, questions, qstory, story, vocabSize, memorySize, enableSoftmax, validationQs, arrayType)
	oldLossPerVal = lossPerVal
	println("Epoch: ", 0)
	#println("Training Loss: ", lossPer, " Training Accuracy: ", acc)
	println("Validation Loss: ", lossPerVal, " Validation Accuracy: ", accVal, "\n")
	for epoch in 1:nEpochs
		if epoch % lrDecayStep == 0
			lr *= lrDecayAmount
		end
		print("Train @time: ")
		@time train(param, questions, qstory, story, vocabSize, memorySize, enableSoftmax, trainQs, arrayType)
		#print("Report @time: ")
		#lossPer, acc = @time report(param, questions, qstory, story, vocabSize, memorySize, enableSoftmax, trainQs, arrayType)
		print("ReportVal @time: ")
		lossPerVal, accVal = @time report(param, questions, qstory, story, vocabSize, memorySize, enableSoftmax, validationQs, arrayType)
		println("Epoch: ", epoch)
		#println("Training Loss: ", lossPer, " Training Accuracy: ", acc)
		println("Validation Loss: ", lossPerVal, " Validation Accuracy: ", accVal, "\n")

		if !enableSoftmax && lossPerVal > oldLossPerVal
			enableSoftmax = true
		end
		oldLossPerVal = lossPerVal
		reportTest(testData, param, vocabSize, memorySize, enableSoftmax, arrayType)
	end
	#reportTest(testData, param, vocabSize, memorySize, enableSoftmax, arrayType)
end

function train(param, questions, qstory, story, vocabSize, memorySize, enableSoftmax, trainQs, arrayType)
	qCount = length(trainQs)
	qRem = rem(qCount, batchSize)
	qMultEnd = div(qCount, batchSize) + 1
	for qMultiplier in 1:qMultEnd
		qStart = batchSize * (qMultiplier - 1) + 1
		qEnd = qStart + batchSize - 1
		if qMultiplier == qMultEnd
			if qRem == 0
				break
			else
				qEnd = qStart + qRem - 1
			end
		end 
		qBounds = trainQs[qStart:qEnd]

		gloss = lossgradient(param, questions, qstory, story, vocabSize, qBounds, memorySize, enableSoftmax, arrayType)
		for k in keys(gloss)
			gnorm = sum(gloss[k] .^ 2);
			gnorm = sqrt(gnorm)
			if gnorm > gradClip
				gloss[k] = (gloss[k] * gradClip) / gnorm;
			end
		end

		for k in keys(param)
			param[k] = param[k] - lr * gloss[k]	
		end
	end
end

function loss(param, questions, qstory, story, vocabSize, qBounds, memorySize, enableSoftmax, arrayType)
	yhat = predict(param, questions, qstory, story, vocabSize, qBounds, memorySize, enableSoftmax
		,weightAdjacent, nHops, isBagOfWords, timeEmbedding, addLinearLayer, addNonLinearity; atype = arrayType)
	answers = questions[1, qBounds]
	totalLoss = 0
	for q in 1:length(answers)
		totalLoss += log(yhat[answers[q], q])
	end
	return -totalLoss # / length(anwers)
end

lossgradient = grad(loss)

function report(param, questions, qstory, story, vocabSize, memorySize, enableSoftmax, qBounds, arrayType)
	yhat = predict(param, questions, qstory, story, vocabSize, qBounds, memorySize, enableSoftmax
		,weightAdjacent, nHops, isBagOfWords, timeEmbedding, addLinearLayer, addNonLinearity; atype = arrayType)
	answers = questions[1, qBounds]
	correct = 0.0
	for i in 1:length(answers)
		correct += answers[i] == indmaxMe(yhat[:, i]) ? 1.0 : 0.0
	end
	totalLoss = 0
	for q in 1:length(answers)
		totalLoss += log(yhat[answers[q], q])
	end
	return -totalLoss / length(answers), correct / length(answers)
end

function reportTest(testData, param, vocabSize, memorySize, enableSoftmax, arrayType)
	println("########## TEST RESULTS ##########")
	for taskID in 1:20
		lossPer, acc = report(param, testData[taskID][3], testData[taskID][2], testData[taskID][1]
			, vocabSize, memorySize, enableSoftmax, collect(1:size(testData[taskID][3], 2)), arrayType)
		println("Test Loss: ", lossPer, " Test Accuracy: ", acc)
	end
end

function indmaxMe(vec)
	idx = 1
	maxEl = vec[1]
	for i in 2:length(vec)
		if vec[i] > maxEl
			maxEl = vec[i]
			idx = i
		end
	end
	return idx
end

main()