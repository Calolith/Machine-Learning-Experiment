include("parseBabi.jl")
include("modelFunctions.jl")

using Knet, AutoGrad, ParseBabi, ModelFunctions

#################### Configurations ####################

	batchSize = 32
	nHops = 3
	nEpochs = 100
	lrDecayStep = 25
	lrDecayAmount = 0.5
	lr = 0.01
	gradClip = 40
	d = 20

	timeEmbedding = true
	isBagOfWords = false
	weightAdjacent = true
	linearStart = false
	addNonLinearity = false
	addLinearLayer = false
	amountOfNoise = 0.0 #TODO
	
########################################################

function main()
	global lr
	baseDir = "data/tasksv11/en/"
	taskID = 1

	# vocab = Dictionary( String -> Integer )
	vocab = createVocab(baseDir, taskID)
	storyTest, qstoryTest, questionsTest = parseBabiTask(vocab, baseDir, taskID; isTest = true)
	# Story = Matrix( Word, Sentence, Story )
	# QStory = Matrix( Word, QuestionIdx )
	# Questions = Matrix( Info, QuestionIdx ) : Info = Vec( Answer, SentencesBefore, StoryIdx ) 
	story, qstory, questions = parseBabiTask(vocab, baseDir, taskID)
	qCount = size(questions, 2)
	allQs = randperm(qCount)
	trainQs = allQs[1:Int64(floor(qCount * 0.9))]
	validationQs = allQs[Int64(floor(qCount * 0.9)) + 1:end]

	memorySize = min(50, size(story, 2))
	vocabSize = length(vocab)
	enableSoftmax = !linearStart
	if weightAdjacent
		param = initweightsAdj(vocabSize, memorySize, nHops, d, timeEmbedding, addLinearLayer)
	else
		param = initweightsLyr(vocabSize, memorySize, d, timeEmbedding, addLinearLayer)
	end

	print("Report @time: ")
	lossPer, acc = @time report(param, questions, qstory, story, vocabSize, memorySize, enableSoftmax, trainQs)
	oldLossPer = lossPer
	print("ReportVal @time: ")
	lossPerVal, accVal = @time report(param, questions, qstory, story, vocabSize, memorySize, enableSoftmax, validationQs)
	oldLossPerVal = lossPerVal
	println("Epoch: ", 0)
	println("Training Loss: ", lossPer, " Training Accuracy: ", acc)
	println("Validation Loss: ", lossPerVal, " Validation Accuracy: ", accVal, "\n")
	for epoch in 1:nEpochs
#		if epoch % lrDecayStep == 0
#			lr *= lrDecayAmount
#		end
		print("Train @time: ")
		@time train(param, questions, qstory, story, vocabSize, memorySize, enableSoftmax, trainQs)
		print("Report @time: ")
		lossPer, acc = @time report(param, questions, qstory, story, vocabSize, memorySize, enableSoftmax, trainQs)
		print("ReportVal @time: ")
		lossPerVal, accVal = @time report(param, questions, qstory, story, vocabSize, memorySize, enableSoftmax, validationQs)
		println("Epoch: ", epoch)
		println("Training Loss: ", lossPer, " Training Accuracy: ", acc)
		println("Validation Loss: ", lossPerVal, " Validation Accuracy: ", accVal, "\n")
		if !enableSoftmax && lossPerVal > oldLossPerVal
			enableSoftmax = true
		end

		reportTest(storyTest, qstoryTest, questionsTest, param, vocabSize, memorySize, enableSoftmax)
		if lossPer > oldLossPer
			lr *= lrDecayAmount
		end
		oldLossPer = lossPer
		oldLossPerVal = lossPerVal

		##################
	end
end

function train(param, questions, qstory, story, vocabSize, memorySize, enableSoftmax, trainQs)
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

		gloss = lossgradient(param, questions, qstory, story, vocabSize, qBounds, memorySize, enableSoftmax)

		for k in keys(gloss)
			gnorm = sum(gloss[k] .^ 2);
			gnorm = sqrt(gnorm)
			if gnorm > gradClip
				gloss[k] = (gloss[k] * gradClip) / gnorm;
			end
		end
#       for k in keys(gloss)
#			gnorm += sum(gloss[k] .^ 2);
#       end
#		gnorm = sqrt(gnorm)
#		if gnorm > 40
#			for k in keys(gloss)
#				gloss[k] = (gloss[k] * 40) / gnorm;
#			end
#		end

		for k in keys(param)
			param[k] = param[k] - lr * gloss[k]	
		end
	end
end

function loss(param, questions, qstory, story, vocabSize, qBounds, memorySize, enableSoftmax)
	yhat = predict(param, questions, qstory, story, vocabSize, qBounds, memorySize, enableSoftmax
		,weightAdjacent, nHops, isBagOfWords, timeEmbedding, addLinearLayer, addNonLinearity)
	answers = questions[1, qBounds]
	totalLoss = 0
	for q in 1:length(answers)
		totalLoss += log(yhat[answers[q], q])
	end
	return -totalLoss # / length(anwers)
end

lossgradient = grad(loss)

function report(param, questions, qstory, story, vocabSize, memorySize, enableSoftmax, qBounds)
	yhat = predict(param, questions, qstory, story, vocabSize, qBounds, memorySize, enableSoftmax
		,weightAdjacent, nHops, isBagOfWords, timeEmbedding, addLinearLayer, addNonLinearity)
	answers = questions[1, qBounds]
	correct = 0.0
	for i in 1:length(answers)
		correct += answers[i] == indmax(yhat[:, i]) ? 1.0 : 0.0
	end
	totalLoss = 0
	for q in 1:length(answers)
		totalLoss += log(yhat[answers[q], q])
	end
	return -totalLoss / length(answers), correct / length(answers)
end

function reportTest(storyTest, qstoryTest, questionsTest, param, vocabSize, memorySize, enableSoftmax)
	lossPer, acc = report(param, questionsTest, qstoryTest, storyTest, vocabSize, memorySize, enableSoftmax, collect(1:size(questionsTest, 2)))
	println("########## TEST RESULTS ##########")
	println("Test Loss: ", lossPer, " Test Accuracy: ", acc)
end

main()