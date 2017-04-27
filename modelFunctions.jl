module ModelFunctions

export softmax, createQuestion, createMemoryMatrix, bagOfWords,
positionEncoding, makeOneHotSentence, initweightsAdj, initweightsLyr, predict

function predict(param, questions, qstory, story, vocabSize, qBounds, memorySize, enableSoftmax
	,weightAdjacent, nHops, isBagOfWords, timeEmbedding, addLinearLayer, addNonLinearity; atype = eval(parse("Array{Float32}")))
	result = zeros(vocabSize, 1)
	result = convert(atype, result)
	if weightAdjacent
		for q in qBounds
			u = createQuestion(param["A1"], qstory[:, q], vocabSize, isBagOfWords, atype)
			sentenceBounds = (max(1, questions[2, q] - memorySize + 1), questions[2, q])
			for hop in 1:nHops
				MA = createMemoryMatrix(param["A" * string(hop)], story, sentenceBounds, questions[3, q], vocabSize, isBagOfWords, atype)
				MC = createMemoryMatrix(param["A" * string(hop + 1)], story, sentenceBounds, questions[3, q], vocabSize, isBagOfWords, atype)
				if timeEmbedding
					sentenceCount = sentenceBounds[2] - sentenceBounds[1] + 1
					MA += param["T" * string(hop)][:, 1:sentenceCount]
					MC += param["T" * string(hop + 1)][:, 1:sentenceCount]
				end
				if enableSoftmax
					p = softmax(transpose(u) * MA, 2)
				else
					p = MA
				end
				o = sum(p .* MC, 2)
				if addLinearLayer
					u = o + param["H"] * u
				else
					u = o + u
				end
				if addNonLinearity
					u = max(0, u)
				end
			end
			yhat = softmax(transpose(param["A" * string(nHops + 1)]) * u, 1)
			result = hcat([result, yhat]...)
		end
	else
		for q in qBounds
			u = createQuestion(param["B"], qstory[:, q], vocabSize, isBagOfWords, atype)
			sentenceBounds = (max(1, questions[2, q] - memorySize + 1), questions[2, q])
			for hop in 1:nHops
				MA = createMemoryMatrix(param["A"], story, sentenceBounds, questions[3, q], vocabSize, isBagOfWords, atype)
				MC = createMemoryMatrix(param["C"], story, sentenceBounds, questions[3, q], vocabSize, isBagOfWords, atype)
				if timeEmbedding
					sentenceCount = sentenceBounds[2] - sentenceBounds[1] + 1
					MA += param["TA"][:, 1:sentenceCount]
					MC += param["TC"][:, 1:sentenceCount]
				end
				if enableSoftmax
					p = softmax(transpose(u) * MA, 2)
				else
					p = MA
				end
				o = sum(p .* MC, 2)
				if addLinearLayer
					u = o + param["H"] * u
				else
					u = o + u
				end
			end
			yhat = softmax(transpose(param["W"]) * u, 1)
			result = hcat([result, yhat]...)
		end
	end
	return result[:, 2:end]
end

function softmax(matrix, side)
	x = exp(matrix - maximum(matrix))
	return x ./ sum(x, side)
end

function createQuestion(matrix, sentence, vocabSize, isBagOfWords, atype)
	if isBagOfWords
		return bagOfWords(matrix, sentence, vocabSize, atype)
	else
		return positionEncoding(matrix, sentence, vocabSize, atype)
	end
end

function createMemoryMatrix(matrix, story, sentenceBounds, storyNum, vocabSize, isBagOfWords, atype)
	if isBagOfWords
		result = bagOfWords(matrix, story[:, sentenceBounds[1], storyNum], vocabSize, atype)
		for j in sentenceBounds[1] + 1:sentenceBounds[2]
			result = hcat([result, bagOfWords(matrix, story[:, j, storyNum], vocabSize, atype)]...)
		end
	else
		result = positionEncoding(matrix, story[:, sentenceBounds[1], storyNum], vocabSize, atype)
		for j in sentenceBounds[1] + 1:sentenceBounds[2]
			result = hcat([result, positionEncoding(matrix, story[:, j, storyNum], vocabSize, atype)]...)
		end
	end
	return result
end

function bagOfWords(matrix, sentence, vocabSize, atype)
	sentenceOhv = makeOneHotSentence(vocabSize, sentence, atype)
	return sum(matrix * sentenceOhv, 2)
end

function positionEncoding(matrix, sentence, vocabSize, atype)
	sentenceOhv = makeOneHotSentence(vocabSize, sentence, atype)
	d = size(matrix, 1)
	J = length(sentence)
	j = [i for i=1:J]
	j = transpose(hcat([j for i=1:d]...))
	k = [i for i=1:d]
	k = hcat([k for i=1:J]...)
	l = (1 - j/J) - (k / d).*(1 - j*(2/J))

	l = convert(atype, l)
	return sum((matrix * sentenceOhv) .* l, 2)
end

function makeOneHotSentence(vocabSize, sentence, atype)
	sentenceOhv = zeros(Int, vocabSize, length(sentence))
	for wordIdx in 1:length(sentence)
		element = sentence[wordIdx]
		if element != 0
			sentenceOhv[element, wordIdx] = 1
		end
	end
	sentenceOhv = convert(atype, sentenceOhv)
	return sentenceOhv
end

function initweightsAdj(vocabSize, memorySize, nHops, d, timeEmbedding, addLinearLayer; atype = eval(parse("Array{Float32}")))
	std = 0.1
	param = Dict()

	for hop in 1:nHops + 1
		param["A" * string(hop)] = randn(d, vocabSize) * std
		if timeEmbedding
			param["T" * string(hop)] = randn(d, memorySize) * std
		end
	end
	if addLinearLayer
		param["H"] = randn(d, d) * std
	end

	for k in keys(param); param[k] = convert(atype, param[k]); end
	return param
end

function initweightsLyr(vocabSize, memorySize, d, timeEmbedding, addLinearLayer; atype = eval(parse("Array{Float32}")))
	std = 0.1
	param = Dict()

	param["B"] = randn(d, vocabSize) * std
	param["A"] = randn(d, vocabSize) * std
	param["C"] = randn(d, vocabSize) * std
	param["W"] = randn(d, vocabSize) * std
	if timeEmbedding
		param["TA"] = randn(d, memorySize) * std
		param["TC"] = randn(d, memorySize) * std
	end
	if addLinearLayer
		param["H"] = randn(d, d) * std
	end

	for k in keys(param); param[k] = convert(atype, param[k]); end
	return param
end

end