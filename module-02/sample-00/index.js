import tf from '@tensorflow/tfjs-node';

async function trainModel(inputXs, outputYs) {
    const model = tf.sequential();

    // First network layer
    // input of 7 positions (normalized age, 3 colors, 3 locations)
    // 80 neuron = train base is low
    // more neuron, more complexity, then more processing
    
    // the ReLU acts as a filter:
    // It's like it lets only the interesting data go through the network
    // If it's ZERO or negative, can throw away, doesn't work
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }))
 
    //Output: 3 neuron
    // one for each category (premium, medium, basic)

    //activation: softmax noramlizes the output in probabilities
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }))

    //Compiling the model
    // optimizer Adam (Adaptive Moment Estimation)
    // It's a modern personal trainer to neural networks
    // Adjust the weight efficiently and smart
    // It learns with historic of mistakes and successes

    //loss: categoricalCrossentropy
    //it compares what the model "thinks" (the scores of each category)
    //with the right answer
    //the premium category will always be [1, 0, 0]

    //As much distant from model's prevision from the right answer
    //greater the error (loss)
    //Example: classify images, recommendation, users categorization
    //anything that the right answer is "just one from many categories"
    
    model.compile({ 
        optimizer: 'adam', 
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    })

    //Training the model
    await model.fit(
        inputXs,
        outputYs,
        {
            verbose: 0, //0 = no log (use callback only), 1 = progress bar, 2 = one line per epoch
            epochs: 100, //number of times the model will see the same data
            shuffle: true, //randomize the order of the data in each epoch, to prevent overfitting
            // callbacks: {
            //     onEpochEnd: (epoch, log) => console.log(
            //         `Epoch: ${epoch}: loss = ${log.loss}`
            //     )
            // }
        }
    )

    return model
}

async function predict(model, person) {
    //transform the JS array to tensor (tfjs)
    const tfInput = tf.tensor2d(person)

    //Do the prediction (output will be a vector of 3 probabilities)
    const pred = model.predict(tfInput)
    const predArray = await pred.array()
    return predArray[0].map((prob, index) => ({ prob, index }))
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels for categories to be predicted (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Labels order
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

//Create the inputs tensor (xs) and output tensor (ys) to train the model
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

//More data = better model
//This way the algorithm can learn more complex patterns
const model = await trainModel(inputXs, outputYs)

const person = { nome: 'ze', idade: 28, cor: 'verde', localizacao: 'Curitiba' }
//Normalizing age from new person utilizing the same train pattern
//example: idade_min = 25, idade_max = 40, so (28-25) / (40-25) = 0.2

const tensorPersonNormalized = [
    [
        0.2, //normalizade age
        1, //azul
        0, //vermelho
        0, //verde
        0, //sp
        1, //rio
        0, //curitiba
    ]
]

const predictions = await predict(model, tensorPersonNormalized)
const results = predictions
    .sort((a,b) => b.prob - a.prob)
    .map(p => `${labelsNomes[p.index]} (${(p.prob*100).toFixed(2)}%)`)
    .join('\n')
console.log(results)