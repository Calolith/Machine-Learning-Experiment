include("parseLangModel.jl")

using Knet, AutoGrad, ParseLangModel, ArgParse

function main()
	config = parse_commandline()
	println("config=",[(k,v) for (k,v) in config]...)
	config[:atype] = eval(parse(config[:atype]))

	vocab = Dict{String, Int}()
	ivocab = Dict{Int, String}()
	vocab["<eos>"] = 1
	ivocab[1] = "<eos>"

	wordsTrain = readData("data/dataptb/ptb.train.txt", vocab, ivocab)
	wordsValid = readData("data/dataptb/ptb.valid.txt", vocab, ivocab)
	wordsTest = readData("data/dataptb/ptb.test.txt", vocab, ivocab)

	costValidOld = 0.0
	param = initweights(config, length(vocab))
	for epoch in 1:config[:epochs]
		@time costTrain = train(param, config, wordsTrain)
		costValid = test(param, config, wordsValid)

		if epoch > 1 && costValid > costValidOld * 0.9999
			config[:lr] = config[:lr] / 1.5
			if config[:lr] < 1e-5
				break
			end
		end
		costValidOld = costValid

		println("Epoch: ", epoch, " Training Loss: ", costTrain, " Validation Perplexity: ", exp(costValid))
	end
	println("Test Perplexity: ", exp(test(param, config, wordsTest)))
end

function test(param, config, words)
	wordCount = length(words)
	N = div(wordCount, config[:batchsize]) + 1
	targets = zeros(Int, config[:batchsize])
	context = zeros(Int, config[:memsize], config[:batchsize])
	initHidden = ones(config[:edim]) * config[:init_hid]
	initHidden = convert(config[:atype], initHidden)
	totalLoss = 0.0
	m = config[:memsize] + 1

	for n in 1:N
		fill!(targets, 0)
		for b in 1:config[:batchsize]
			targets[b] = words[m]
			context[:, b] = words[m - config[:memsize]:m - 1]
			m += 1
			if m > wordCount - 1;
				m = config[:memsize] + 1
			end
		end
		totalLoss += loss(param, config, context, targets, initHidden)
	end
	return totalLoss / N
end

function train(param, config, words)
	wordCount = length(words)
	N = div(wordCount, config[:batchsize]) + 1
	targets = zeros(Int, config[:batchsize])
	context = zeros(Int, config[:memsize], config[:batchsize])
	initHidden = ones(config[:edim]) * config[:init_hid]
	initHidden = convert(config[:atype], initHidden)
	totalLoss = 0.0

	for n in 1:N
		fill!(targets, 0)
		for b in 1:config[:batchsize]
			m = rand(config[:memsize] + 1:wordCount)
			targets[b] = words[m]
			context[:, b] = words[m - config[:memsize]:m - 1]
		end

		totalLoss += loss(param, config, context, targets, initHidden)
		gloss = lossgrad(param, config, context, targets, initHidden)
		gnorm = 0.0
        for k in keys(gloss)
			gnorm += sum(gloss[k] .^ 2);
        end
		gnorm = sqrt(gnorm)
		if gnorm > config[:maxgradnorm]
			for k in keys(gloss)
				gloss[k] = (gloss[k] * config[:maxgradnorm]) / gnorm;
			end
		end

		for k in keys(param)
			param[k] = param[k] - config[:lr] * gloss[k]	
		end
	end
	return totalLoss / N
end

function loss(param, config, context, targets, initHidden)
	totalLoss = 0.0
	for b in 1:size(context, 2)
		M = param[:A][:, context[:, b]]
		M += param[:TA]
		C = param[:C][:, context[:, b]]
		C += param[:TC]
		u = copy(initHidden)

		for hop in 1:config[:nhop]
			p = softmax(transpose(u) * M, 2)
			o = C * transpose(p)
			u = o + param[:H] * u
		end

		yhat = softmax(param[:W] * u, 1)
		totalLoss += log(yhat[targets[b]])
	end

	return -totalLoss / length(targets)
end

lossgrad = grad(loss)

function softmax(matrix, side)
	x = exp(matrix - maximum(matrix) + 1)
	return x ./ sum(x, side)
end

function initweights(config, vocabSize)
	std = config[:init_std]
	d = config[:edim]
	param = Dict()

	param[:A] = randn(d, vocabSize) * std
	param[:C] = randn(d, vocabSize) * std
	param[:TA] = randn(d, config[:memsize]) * std
	param[:TC] = randn(d, config[:memsize]) * std
	param[:H] = randn(d, d) * std 
	param[:W] = randn(vocabSize, d) * std
 
	for k in keys(param); param[k] = convert(config[:atype], param[k]); end
	return param
end

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
    	("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
        ("--edim"; arg_type=Int; default=150; help="internal state dimension")
        ("--lindim";  arg_type=Int; default=75; help="linear part of the state")
        ("--init_std"; arg_type=Float64; default=0.05; help="weight initialization std")
        ("--init_hid"; arg_type=Float64; default=0.1; help="initial internal state value")
        ("--lr"; arg_type=Float64; default=0.01; help="initial learning rate")
        ("--maxgradnorm"; arg_type=Int; default=50; help="maximum gradient norm")
        ("--memsize"; arg_type=Int; default=100; help="memory size")
        ("--nhop"; arg_type=Int; default=6; help="number of hops")
        ("--batchsize"; arg_type=Int; default=128; help="batch size")
        ("--epochs"; arg_type=Int; default=100; help="number of epochs")  
    end
    return parse_args(s;as_symbols = true)        
end

main()