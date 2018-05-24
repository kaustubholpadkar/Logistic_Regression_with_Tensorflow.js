var changeColor;

// Data
X1 = [];
X2 = [];
Y  = [];

var W1, W2, B

const w1 = tf.variable(tf.scalar(Math.random()))
const w2 = tf.variable(tf.scalar(Math.random()))
const b =  tf.variable(tf.scalar(Math.random()))

const learningRate = 0.9
const optimizer = tf.train.sgd(learningRate)

var type = 1;

var flag = false;
var isMobile;

function setup () {
  if( /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) ) {
    createCanvas(windowWidth, windowWidth - 200)
    isMobile = true
    createElement("br")
    changeColor = createButton("Change Color");
    changeColor.touchMoved(swap);
    createSpan("_");
  } else {
    isMobile = false
    createCanvas(windowWidth, windowHeight)
  }
}

function draw () {
  background(51)
  intro()
  plotData()

  if (X1.length) {
    tf.tidy(() => {
      const x1 = tf.tensor(X1, [X1.length, 1])
      const x2 = tf.tensor(X2, [X2.length, 1])
      const ys = tf.tensor(Y, [Y.length, 1])

      train(x1, x2, ys)

      W1 = w1.dataSync()[0]
      W2 = w2.dataSync()[0]
      B = b.dataSync()[0]
    });
    drawLine()
  }

  // Check Memory Leak
  // console.log(tf.memory().numTensors);
}

function intro () {
  fill(250)
  noStroke()
  textFont('monospace')
  textSize(25)
  text("Tensorflow.js : Logistic Regression", 15, 40)
  textSize(20)
  if (isMobile) {
    let instruction = "Tap on the Screen to Insert Data Points..."
    text(instruction, 15, windowHeight - 60)
  } else {
    let instruction = "Tap on the Screen to Insert Data Points...         Press Enter to change the class"
    text(instruction, 15, windowHeight - 30)
  }

  fill(100)
  textSize(15)
  text("Author : Kaustubh Olpadkar", windowWidth - 270, 40)
  noFill();
  noStroke();
}


function mouseClicked () {
  if (mouseX < width && mouseY < height) {
    let normX1 = map(mouseX, 0, width, 0, 1)
    let normX2 = map(mouseY, 0, height, 0, 1)

    X1.push(normX1);
    X2.push(normX2);

    Y.push(type);
  }
}

function swap() {
  type = 1 - type;
}

function predict(x1, x2) {
  return tf.sigmoid(w1.mul(x1).add(w2.mul(x2)).add(b))
}

function loss(predictions, labels) {
  return tf.scalar(0).sub(tf.mean((labels.mul(tf.log(predictions))).add(((tf.scalar(1).sub(labels)).mul(tf.log(tf.scalar(1).sub(predictions)))))))
}

function train (x1, x2, ys, numIterations = 1) {
  for (let iter = 0; iter < numIterations; iter++) {
    optimizer.minimize(() => loss(predict(x1, x2), ys));
  }
}

function plotData () {
  noStroke();
  for (var i = 0; i < X1.length; i++) {
    Y[i] == 1 ? fill(255, 0, 0) : fill(0, 0, 255);
    let denormX = Math.floor(map(X1[i], 0, 1, 0, width))
    let denormY = Math.floor(map(X2[i], 0, 1, 0, height))
    ellipse(denormX, denormY, 10);
  }
  noFill();
}

function drawLine () {
  let m = - (W1 / W2);
  let c = - (B / W2);

  let x1 = 0.0
  let y1 = m * x1 + c;
  let x2 = 1.0
  let y2 = m * x2 + c;

  let denormX1 = Math.floor(map(x1, 0, 1, 0, width))
  let denormY1 = Math.floor(map(y1, 0, 1, 0, height))
  let denormX2 = Math.floor(map(x2, 0, 1, 0, width))
  let denormY2 = Math.floor(map(y2, 0, 1, 0, height))

  stroke(255);
  line(denormX1, denormY1, denormX2, denormY2);
}

function keyPressed () {
  if (keyCode === ENTER) {
    swap()
  } else if (keyCode === ESCAPE) {
    flag = true;
  }
}
