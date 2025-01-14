// 🎲 Time Oracle - Where randomness meets intention
const spacebarDot = document.getElementById('spacebar-dot');
const randomDot = document.getElementById('random-dot');
const ctx = document.getElementById('timeChart').getContext('2d');
const humanDurationCtx = document.getElementById('humanDurationChart').getContext('2d');
const randomDurationCtx = document.getElementById('randomDurationChart').getContext('2d');
const correlationCtx = document.getElementById('correlationChart').getContext('2d');
const predictionDot = document.getElementById('prediction-dot');
const predictionDurationCtx = document.getElementById('predictionDurationChart').getContext('2d');
const predictionAnalysisCtx = document.getElementById('predictionChart').getContext('2d');

// State tracking
let spacebarState = false;
let randomState = false;
const THREE_MINUTES = 3 * 60 * 1000;
const UPDATE_INTERVAL = 100;
const maxDataPoints = THREE_MINUTES / UPDATE_INTERVAL;
const startTime = Date.now();

// Duration tracking
let lastSpacebarPress = 0;
let lastSpacebarRelease = 0;
let lastRandomOn = 0;
let lastRandomOff = 0;
const maxDurationPoints = 100;
const spacebarPressDurations = [];
const spacebarIntervals = [];
const randomPressDurations = [];
const randomIntervals = [];
const predictionPressDurations = [];
const predictionIntervals = [];

// Histogram bins configuration
const BIN_COUNT = 20;
let pressBins = Array(BIN_COUNT).fill(0);
let intervalBins = Array(BIN_COUNT).fill(0);
let randomPressBins = Array(BIN_COUNT).fill(0);
let randomIntervalBins = Array(BIN_COUNT).fill(0);
let predictionPressBins = Array(BIN_COUNT).fill(0);
let predictionIntervalBins = Array(BIN_COUNT).fill(0);

// Helper function to calculate histogram bins and labels
function calculateHistogram(data, binCount) {
    if (data.length === 0) return { bins: Array(binCount).fill(0), labels: Array(binCount).fill('') };
    
    // Remove any extreme outliers (more than 3 standard deviations)
    const mean = data.reduce((a, b) => a + b, 0) / data.length;
    const stdDev = Math.sqrt(data.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / data.length);
    const filteredData = data.filter(x => Math.abs(x - mean) <= 3 * stdDev);
    
    if (filteredData.length === 0) return { bins: Array(binCount).fill(0), labels: Array(binCount).fill('') };
    
    const min = Math.min(...filteredData);
    const max = Math.max(...filteredData);
    const binWidth = (max - min) / binCount;
    const bins = Array(binCount).fill(0);
    const labels = Array(binCount).fill('');
    
    // Create labels first
    for (let i = 0; i < binCount; i++) {
        const start = min + (i * binWidth);
        const end = start + binWidth;
        labels[i] = `${Math.round(start)}-${Math.round(end)}ms`;
    }
    
    // Fill the bins
    filteredData.forEach(value => {
        const binIndex = Math.min(Math.floor((value - min) / binWidth), binCount - 1);
        bins[binIndex]++;
    });
    
    return { bins, labels };
}

// Calculate correlation between two distributions
function calculateCorrelation(dist1, dist2) {
    if (dist1.length === 0 || dist2.length === 0) return 0;
    if (dist1.every(x => x === 0) || dist2.every(x => x === 0)) return 0;
    
    // Normalize the distributions
    const sum1 = dist1.reduce((a, b) => a + b, 0);
    const sum2 = dist2.reduce((a, b) => a + b, 0);
    if (sum1 === 0 || sum2 === 0) return 0;
    
    const norm1 = dist1.map(v => v / sum1);
    const norm2 = dist2.map(v => v / sum2);
    
    // Calculate correlation coefficient
    const mean1 = norm1.reduce((a, b) => a + b) / norm1.length;
    const mean2 = norm2.reduce((a, b) => a + b) / norm2.length;
    
    let numerator = 0;
    let denom1 = 0;
    let denom2 = 0;
    
    for (let i = 0; i < norm1.length; i++) {
        const diff1 = norm1[i] - mean1;
        const diff2 = norm2[i] - mean2;
        numerator += diff1 * diff2;
        denom1 += diff1 * diff1;
        denom2 += diff2 * diff2;
    }
    
    if (denom1 === 0 || denom2 === 0) return 0;
    return numerator / Math.sqrt(denom1 * denom2);
}

// Initialize timestamps for state chart
const timestamps = Array(maxDataPoints).fill(0).map((_, i) => {
    const time = new Date(startTime - (maxDataPoints - i - 1) * UPDATE_INTERVAL);
    return time.toLocaleTimeString('en-US', { 
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        fractionalSecondDigits: 1
    });
});
const spacebarData = Array(maxDataPoints).fill(null);
const randomData = Array(maxDataPoints).fill(null);

// State Chart initialization (remains line chart)
const stateChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: timestamps,
        datasets: [
            {
                label: 'Spacebar Dot',
                data: spacebarData,
                borderColor: 'rgb(255, 99, 132)',
                tension: 0.1,
                pointRadius: 0,
                spanGaps: true
            },
            {
                label: 'Random Dot',
                data: randomData,
                borderColor: 'rgb(54, 162, 235)',
                tension: 0.1,
                pointRadius: 0,
                spanGaps: true
            },
            {
                label: 'Predicted Press',
                data: Array(maxDataPoints).fill(null),
                borderColor: 'rgb(255, 193, 7)',
                backgroundColor: 'rgba(255, 193, 7, 0.2)',
                borderDash: [5, 5],
                tension: 0.1,
                pointRadius: 0
            }
        ]
    },
    options: {
        responsive: true,
        scales: {
            y: {
                min: -0.1,
                max: 1.1,
                ticks: {
                    callback: value => value === 0 ? 'Off' : value === 1 ? 'On' : ''
                }
            },
            x: {
                ticks: {
                    maxRotation: 0,
                    autoSkip: true,
                    maxTicksLimit: 10
                }
            }
        },
        animation: {
            duration: 0
        }
    }
});

// Human Duration Chart initialization
const humanDurationChart = new Chart(humanDurationCtx, {
    type: 'bar',
    data: {
        labels: Array(BIN_COUNT).fill(''),
        datasets: [
            {
                label: 'Press Duration',
                data: pressBins,
                backgroundColor: 'rgba(255, 99, 132, 0.5)',
                borderColor: 'rgb(255, 99, 132)',
                borderWidth: 1
            },
            {
                label: 'Interval Between Presses',
                data: intervalBins,
                backgroundColor: 'rgba(255, 150, 150, 0.5)',
                borderColor: 'rgb(255, 150, 150)',
                borderWidth: 1
            }
        ]
    },
    options: {
        responsive: true,
        scales: {
            y: {
                beginAtZero: true,
                title: {
                    display: true,
                    text: 'Frequency'
                }
            },
            x: {
                title: {
                    display: true,
                    text: 'Duration (ms)'
                },
                ticks: {
                    maxRotation: 45,
                    minRotation: 45
                }
            }
        },
        plugins: {
            title: {
                display: true,
                text: '🎯 Human Timing Distribution'
            }
        },
        animation: {
            duration: 0
        }
    }
});

// Random Duration Chart initialization
const randomDurationChart = new Chart(randomDurationCtx, {
    type: 'bar',
    data: {
        labels: Array(BIN_COUNT).fill(''),
        datasets: [
            {
                label: 'Green Duration',
                data: randomPressBins,
                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                borderColor: 'rgb(54, 162, 235)',
                borderWidth: 1
            },
            {
                label: 'Interval Between Green',
                data: randomIntervalBins,
                backgroundColor: 'rgba(100, 200, 255, 0.5)',
                borderColor: 'rgb(100, 200, 255)',
                borderWidth: 1
            }
        ]
    },
    options: {
        responsive: true,
        scales: {
            y: {
                beginAtZero: true,
                title: {
                    display: true,
                    text: 'Frequency'
                }
            },
            x: {
                title: {
                    display: true,
                    text: 'Duration (ms)'
                },
                ticks: {
                    maxRotation: 45,
                    minRotation: 45
                }
            }
        },
        plugins: {
            title: {
                display: true,
                text: '🎲 Random Event Distribution'
            }
        },
        animation: {
            duration: 0
        }
    }
});

// Correlation Chart initialization
const correlationChart = new Chart(correlationCtx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'Press Duration Correlation',
                data: [],
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                tension: 0.1
            },
            {
                label: 'Interval Correlation',
                data: [],
                borderColor: 'rgb(153, 102, 255)',
                backgroundColor: 'rgba(153, 102, 255, 0.2)',
                tension: 0.1
            },
            {
                label: 'Human-Prediction Correlation',
                data: [],
                borderColor: 'rgb(255, 193, 7)',
                backgroundColor: 'rgba(255, 193, 7, 0.2)',
                tension: 0.1
            }
        ]
    },
    options: {
        responsive: true,
        scales: {
            y: {
                min: -1,
                max: 1,
                title: {
                    display: true,
                    text: 'Correlation Coefficient'
                }
            }
        },
        plugins: {
            title: {
                display: true,
                text: '🤔 Human vs Random Correlation'
            }
        },
        animation: {
            duration: 0
        }
    }
});

// Prediction distribution chart
const predictionDurationChart = new Chart(predictionDurationCtx, {
    type: 'bar',
    data: {
        labels: Array(BIN_COUNT).fill(''),
        datasets: [
            {
                label: 'Predicted Press Duration',
                data: predictionPressBins,
                backgroundColor: 'rgba(255, 193, 7, 0.5)',
                borderColor: 'rgb(255, 193, 7)',
                borderWidth: 1
            },
            {
                label: 'Predicted Interval',
                data: predictionIntervalBins,
                backgroundColor: 'rgba(255, 220, 100, 0.5)',
                borderColor: 'rgb(255, 220, 100)',
                borderWidth: 1
            }
        ]
    },
    options: {
        responsive: true,
        scales: {
            y: {
                beginAtZero: true,
                title: {
                    display: true,
                    text: 'Frequency'
                }
            },
            x: {
                title: {
                    display: true,
                    text: 'Duration (ms)'
                },
                ticks: {
                    maxRotation: 45,
                    minRotation: 45
                }
            }
        },
        plugins: {
            title: {
                display: true,
                text: '🔮 Prediction Distribution'
            }
        },
        animation: {
            duration: 0
        }
    }
});

// Prediction analysis chart
const predictionAnalysisChart = new Chart(predictionAnalysisCtx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'Prediction Error (ms)',
                data: [],
                borderColor: 'rgb(255, 193, 7)',
                backgroundColor: 'rgba(255, 193, 7, 0.2)',
                tension: 0.1
            },
            {
                label: 'Running Accuracy',
                data: [],
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                tension: 0.1,
                yAxisID: 'accuracy'
            }
        ]
    },
    options: {
        responsive: true,
        scales: {
            y: {
                title: {
                    display: true,
                    text: 'Error (ms)'
                }
            },
            accuracy: {
                position: 'right',
                title: {
                    display: true,
                    text: 'Accuracy %'
                },
                min: 0,
                max: 100
            }
        },
        plugins: {
            title: {
                display: true,
                text: '🎯 Prediction Performance'
            }
        },
        animation: {
            duration: 0
        }
    }
});

// Helper function to add duration data
function addDurationData(array, value) {
    array.push(value);
    if (array.length > maxDurationPoints) {
        array.shift();
    }
}

// Prediction tracking
let lastPrediction = 0;
let predictionAccuracy = [];
const PREDICTION_WINDOW = 200; // ms tolerance for prediction accuracy
const MIN_SAMPLES_FOR_PREDICTION = 5;
const PREDICTION_HORIZON = 3000; // How far ahead to predict (ms)

// Add after the initial constants
const MIN_HUMAN_DURATION = 20; // Minimum human press duration in ms
const MIN_PREDICTION_DURATION = 20; // Minimum prediction duration in ms
const leadTimeSlider = document.getElementById('lead-time');
const leadTimeDisplay = document.getElementById('lead-time-display');

// Update lead time display
leadTimeSlider.addEventListener('input', (e) => {
    const value = e.target.value;
    leadTimeDisplay.textContent = `${value}ms`;
});

// Add state space chart initialization
const stateSpaceChart = new Chart(document.getElementById('stateSpaceChart').getContext('2d'), {
    type: 'scatter',
    data: {
        datasets: [{
            label: 'Phase Space',
            data: [],
            backgroundColor: 'rgba(255, 193, 7, 0.5)',
            borderColor: 'rgb(255, 193, 7)',
            showLine: true,
            tension: 0.1
        },
        {
            label: 'Frequency Components',
            data: [],
            backgroundColor: 'rgba(75, 192, 192, 0.5)',
            borderColor: 'rgb(75, 192, 192)',
            showLine: false
        }]
    },
    options: {
        responsive: true,
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Phase / Frequency (Hz)'
                }
            },
            y: {
                title: {
                    display: true,
                    text: 'Strength / Magnitude'
                }
            }
        },
        plugins: {
            title: {
                display: true,
                text: '🧠 Prediction State Space'
            }
        },
        animation: {
            duration: 0
        }
    }
});

// Prediction model
class PredictionModel {
    constructor() {
        this.recentIntervals = [];
        this.recentDurations = [];
        // Use a power of 2 for FFT size
        this.FFT_SIZE = 1024;  // 2^10, a good size for our purpose
        this.maxSamples = this.FFT_SIZE;
        this.lastPredictionTime = 0;
        this.nextPredictedPress = 0;
        this.lastPredictionDuration = 0;
        this.predictionErrors = [];
        this.predictionState = false;
        this.confidence = 0;
        this.baseInterval = 1000;
        this.phase = 0;
        this.harmonics = [];
        this.rhythmBuffer = new Array(this.FFT_SIZE).fill(0);
        this.bufferIndex = 0;
        this.lastTimestamp = 0;
        
        // Window function for better FFT results
        this.window = new Array(this.FFT_SIZE);
        for (let i = 0; i < this.FFT_SIZE; i++) {
            // Hanning window
            this.window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (this.FFT_SIZE - 1)));
        }
    }

    addInterval(interval) {
        this.recentIntervals.push(interval);
        if (this.recentIntervals.length > this.maxSamples) {
            this.recentIntervals.shift();
        }
        // Add to rhythm buffer with windowing
        this.rhythmBuffer[this.bufferIndex] = interval * this.window[this.bufferIndex];
        this.bufferIndex = (this.bufferIndex + 1) % this.FFT_SIZE;
    }

    addDuration(duration) {
        this.recentDurations.push(duration);
        if (this.recentDurations.length > this.maxSamples) {
            this.recentDurations.shift();
        }
    }

    // FFT implementation
    fft(buffer) {
        const n = buffer.length;
        
        // Ensure buffer length is power of 2
        if ((n & (n - 1)) !== 0) {
            throw new Error('FFT length must be power of 2');
        }

        // Base case
        if (n === 1) {
            return [{ real: buffer[0], imag: 0 }];
        }

        // Split into even and odd
        const even = new Array(n/2);
        const odd = new Array(n/2);
        for (let i = 0; i < n/2; i++) {
            even[i] = buffer[2*i];
            odd[i] = buffer[2*i + 1];
        }

        // Recursive FFT on even and odd parts
        const evenFFT = this.fft(even);
        const oddFFT = this.fft(odd);

        // Combine results
        const result = new Array(n);
        for (let k = 0; k < n/2; k++) {
            const theta = -2 * Math.PI * k / n;
            const tk = {
                real: Math.cos(theta),
                imag: Math.sin(theta)
            };

            // Even part
            const evenPart = evenFFT[k];
            
            // Odd part multiplied by twiddle factor
            const oddPart = oddFFT[k];
            const oddTimesT = {
                real: oddPart.real * tk.real - oddPart.imag * tk.imag,
                imag: oddPart.real * tk.imag + oddPart.imag * tk.real
            };

            // Sum for first half
            result[k] = {
                real: evenPart.real + oddTimesT.real,
                imag: evenPart.imag + oddTimesT.imag
            };

            // Difference for second half
            result[k + n/2] = {
                real: evenPart.real - oddTimesT.real,
                imag: evenPart.imag - oddTimesT.imag
            };
        }

        return result;
    }

    updateRhythmModel(timestamp) {
        if (this.lastTimestamp === 0) {
            this.lastTimestamp = timestamp;
            return;
        }

        // Update rhythm buffer with time since last update
        const delta = timestamp - this.lastTimestamp;
        this.rhythmBuffer[this.bufferIndex] = delta * this.window[this.bufferIndex];
        this.bufferIndex = (this.bufferIndex + 1) % this.FFT_SIZE;
        this.lastTimestamp = timestamp;

        // Create a copy of the buffer for FFT
        const paddedBuffer = [...this.rhythmBuffer];
        
        // Zero-pad to FFT_SIZE if needed (should already be correct size)
        while (paddedBuffer.length < this.FFT_SIZE) {
            paddedBuffer.push(0);
        }

        // Perform FFT analysis when we have enough data
        if (this.recentIntervals.length >= MIN_SAMPLES_FOR_PREDICTION) {
            try {
                const fftResult = this.fft(paddedBuffer);
                
                // Find dominant frequencies, excluding very low frequencies
                this.harmonics = fftResult.map((val, idx) => ({
                    frequency: idx / (this.FFT_SIZE * UPDATE_INTERVAL / 1000),  // Convert to Hz
                    magnitude: Math.sqrt(val.real * val.real + val.imag * val.imag),
                    phase: Math.atan2(val.imag, val.real)
                }))
                .slice(1, this.FFT_SIZE/4)  // Remove DC and high frequencies
                .filter(h => h.frequency >= 0.1 && h.frequency <= 10)  // Only keep reasonable frequencies (0.1-10 Hz)
                .sort((a, b) => b.magnitude - a.magnitude)
                .slice(0, 3);  // Keep top 3 harmonics
            } catch (e) {
                console.warn('FFT failed:', e);
                this.harmonics = [];
            }
        }
    }

    predictNextPress(currentTime) {
        this.updateRhythmModel(currentTime);

        if (this.recentIntervals.length < MIN_SAMPLES_FOR_PREDICTION) {
            this.confidence = 0;
            return null;
        }

        // Use harmonics to predict next press
        let prediction = 0;
        let totalWeight = 0;

        this.harmonics.forEach(harmonic => {
            if (harmonic.frequency > 0) {  // Avoid DC component
                const weight = harmonic.magnitude;
                const period = 1000 / harmonic.frequency;  // Convert to milliseconds
                
                // Calculate phase-aware prediction
                const t = currentTime / 1000;
                const currentPhase = (t * harmonic.frequency) % 1;
                const nextBeat = period * (1 - currentPhase);
                
                prediction += nextBeat * weight;
                totalWeight += weight;
            }
        });

        // Weighted average of harmonic predictions
        if (totalWeight > 0) {
            prediction /= totalWeight;
            this.nextPredictedPress = currentTime + Math.max(prediction, MIN_PREDICTION_DURATION);
            this.confidence = Math.min(1, totalWeight / (this.harmonics[0]?.magnitude || 1));
            
            // Add some randomness based on confidence
            const jitter = (1 - this.confidence) * 50;  // Max 50ms jitter when confidence is low
            this.nextPredictedPress += (Math.random() - 0.5) * jitter;
        }

        // Update base interval for visualization
        this.baseInterval = this.harmonics[0]?.frequency ? 1000 / this.harmonics[0].frequency : 1000;

        return this.nextPredictedPress;
    }

    checkAccuracy(actualPressTime) {
        if (!this.nextPredictedPress) return null;
        
        const error = actualPressTime - this.nextPredictedPress;
        const absError = Math.abs(error);
        const isAccurate = absError < PREDICTION_WINDOW;
        
        // Add to rolling accuracy and errors
        predictionAccuracy.push(isAccurate);
        this.predictionErrors.push(error);
        if (predictionAccuracy.length > 50) {
            predictionAccuracy.shift();
            this.predictionErrors.shift();
        }
        
        return error;
    }

    updatePredictionState(currentTime) {
        const leadTime = parseInt(leadTimeSlider.value);
        
        // Calculate combined phase from harmonics
        let phaseSum = 0;
        let magnitudeSum = 0;
        
        this.harmonics.forEach(harmonic => {
            const t = currentTime / 1000;  // Convert to seconds
            const freq = harmonic.frequency;
            if (freq > 0) {  // Avoid DC component
                const periodPhase = (t * freq) % 1;  // Normalized phase in current period
                const contribution = harmonic.magnitude * Math.sin(2 * Math.PI * periodPhase);
                phaseSum += contribution;
                magnitudeSum += harmonic.magnitude;
            }
        });

        // Normalize the combined phase
        const normalizedPhase = magnitudeSum > 0 ? phaseSum / magnitudeSum : 0;
        
        // Calculate time to next predicted press
        const timeToPredict = this.nextPredictedPress - currentTime;
        
        // Calculate prediction strength based on timing and confidence
        const timingFactor = Math.exp(-Math.abs(timeToPredict - leadTime) / (leadTime / 2));
        const predictionStrength = this.confidence * timingFactor;
        
        // Combine phase and prediction strength for pulsing effect
        const pulseEffect = (normalizedPhase + 1) / 2;  // Convert from [-1,1] to [0,1]
        const brightness = 0.3 + (0.7 * pulseEffect * predictionStrength);
        
        // Update prediction dot color using HSL
        predictionDot.style.backgroundColor = `hsl(60, 100%, ${brightness * 50}%)`;
        
        // Update state space visualization
        this.updateStateSpace(currentTime, normalizedPhase, predictionStrength);
        
        this.predictionState = predictionStrength > 0.5;
        return this.predictionState;
    }

    updateStateSpace(currentTime, phase, strength) {
        // Add state space point to visualization
        if (stateSpaceChart.data.datasets[0].data.length > 100) {
            stateSpaceChart.data.datasets[0].data.shift();
            stateSpaceChart.data.datasets[1].data.shift();
        }

        // Plot phase vs prediction strength
        stateSpaceChart.data.datasets[0].data.push({
            x: phase,
            y: strength
        });

        // Plot dominant frequencies
        const freqPoint = {
            x: this.harmonics[0]?.frequency || 0,
            y: this.harmonics[0]?.magnitude || 0
        };
        stateSpaceChart.data.datasets[1].data.push(freqPoint);

        stateSpaceChart.update();
    }

    getAccuracy() {
        if (predictionAccuracy.length === 0) return 0;
        return predictionAccuracy.reduce((a, b) => a + (b ? 1 : 0), 0) / predictionAccuracy.length;
    }
}

const predictionModel = new PredictionModel();

// Key event handlers
document.addEventListener('keydown', (e) => {
    if (e.code === 'KeyA' && !spacebarState) {
        e.preventDefault();
        const now = Date.now();
        
        // Check prediction accuracy
        predictionModel.checkAccuracy(now);
        
        if (lastSpacebarRelease > 0) {
            const interval = Math.max(now - lastSpacebarRelease, MIN_HUMAN_DURATION);
            if (interval < 10000) {
                addDurationData(spacebarIntervals, interval);
                predictionModel.addInterval(interval);
            }
        }
        lastSpacebarPress = now;
        spacebarState = true;
        spacebarDot.style.backgroundColor = 'green';
    }
});

document.addEventListener('keyup', (e) => {
    if (e.code === 'KeyA') {
        e.preventDefault();
        const now = Date.now();
        const duration = Math.max(now - lastSpacebarPress, MIN_HUMAN_DURATION);
        if (duration < 10000) {
            addDurationData(spacebarPressDurations, duration);
            predictionModel.addDuration(duration);
        }
        lastSpacebarRelease = now;
        spacebarState = false;
        spacebarDot.style.backgroundColor = 'red';
    }
});

// Random dot behavior
function updateRandomDot() {
    const now = Date.now();
    const newState = Math.random() < 0.3;
    
    if (newState !== randomState) {
        if (newState) {
            if (lastRandomOff > 0) {
                const interval = now - lastRandomOff;
                // Only record intervals that make sense (less than 10 seconds)
                if (interval < 10000) {
                    addDurationData(randomIntervals, interval);
                }
            }
            lastRandomOn = now;
        } else {
            const duration = now - lastRandomOn;
            // Only record durations that make sense (less than 10 seconds)
            if (duration < 10000) {
                addDurationData(randomPressDurations, duration);
            }
            lastRandomOff = now;
        }
        randomState = newState;
        randomDot.style.backgroundColor = randomState ? 'green' : 'red';
    }
    
    setTimeout(updateRandomDot, Math.random() * 900 + 100);
}

let lastUpdateTime = 0;
// Update charts
function updateCharts() {
    const now = Date.now();
    if (now - lastUpdateTime < UPDATE_INTERVAL) {
        requestAnimationFrame(updateCharts);
        return;
    }
    lastUpdateTime = now;

    // Update prediction state
    predictionModel.updatePredictionState(now);

    const timeStr = new Date(now).toLocaleTimeString('en-US', { 
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        fractionalSecondDigits: 1
    });

    // Update prediction
    if (!spacebarState && 
        (now - predictionModel.lastPredictionTime > UPDATE_INTERVAL) && 
        (now - lastSpacebarRelease > UPDATE_INTERVAL)) {
        predictionModel.predictNextPress(now);
    }

    // Update state chart
    timestamps.shift();
    spacebarData.shift();
    randomData.shift();
    stateChart.data.datasets[2].data.shift();
    
    timestamps.push(timeStr);
    spacebarData.push(spacebarState ? 1 : 0);
    randomData.push(randomState ? 1 : 0);

    // Update prediction visualization
    const timeToPredict = predictionModel.nextPredictedPress - now;
    const predictionValue = timeToPredict > 0 && timeToPredict < PREDICTION_HORIZON ? 0.5 : null;
    stateChart.data.datasets[2].data.push(predictionValue);

    // Update histogram data
    const humanPress = calculateHistogram(spacebarPressDurations, BIN_COUNT);
    const humanInterval = calculateHistogram(spacebarIntervals, BIN_COUNT);
    const randomPress = calculateHistogram(randomPressDurations, BIN_COUNT);
    const randomInterval = calculateHistogram(randomIntervals, BIN_COUNT);

    // Update chart titles with prediction accuracy
    const accuracy = predictionModel.getAccuracy() * 100;
    humanDurationChart.options.plugins.title.text = 
        `🎯 Human Timing Distribution (Prediction Accuracy: ${accuracy.toFixed(1)}%)`;

    // Update human duration chart
    humanDurationChart.data.labels = humanPress.labels;
    humanDurationChart.data.datasets[0].data = humanPress.bins;
    humanDurationChart.data.datasets[1].data = humanInterval.bins;

    // Update random duration chart
    randomDurationChart.data.labels = randomPress.labels;
    randomDurationChart.data.datasets[0].data = randomPress.bins;
    randomDurationChart.data.datasets[1].data = randomInterval.bins;

    // Update prediction distribution chart
    const predictionPress = calculateHistogram(predictionPressDurations, BIN_COUNT);
    const predictionInterval = calculateHistogram(predictionIntervals, BIN_COUNT);

    predictionDurationChart.data.labels = predictionPress.labels;
    predictionDurationChart.data.datasets[0].data = predictionPress.bins;
    predictionDurationChart.data.datasets[1].data = predictionInterval.bins;

    // Update correlation data
    if (correlationChart.data.labels.length > 50) {
        correlationChart.data.labels.shift();
        correlationChart.data.datasets.forEach(dataset => dataset.data.shift());
    }
    
    correlationChart.data.labels.push(timeStr);
    correlationChart.data.datasets[0].data.push(
        calculateCorrelation(humanPress.bins, randomPress.bins)
    );
    correlationChart.data.datasets[1].data.push(
        calculateCorrelation(humanInterval.bins, randomInterval.bins)
    );
    correlationChart.data.datasets[2].data.push(
        calculateCorrelation(humanPress.bins, predictionPress.bins)
    );

    // Update prediction analysis
    if (predictionAnalysisChart.data.labels.length > 50) {
        predictionAnalysisChart.data.labels.shift();
        predictionAnalysisChart.data.datasets.forEach(dataset => dataset.data.shift());
    }

    predictionAnalysisChart.data.labels.push(timeStr);
    predictionAnalysisChart.data.datasets[0].data.push(
        predictionModel.predictionErrors[predictionModel.predictionErrors.length - 1] || 0
    );
    predictionAnalysisChart.data.datasets[1].data.push(
        predictionModel.getAccuracy() * 100
    );

    // Update all charts
    stateChart.update();
    humanDurationChart.update();
    randomDurationChart.update();
    predictionDurationChart.update();
    correlationChart.update();
    predictionAnalysisChart.update();
    requestAnimationFrame(updateCharts);
}

// Start the animations
updateRandomDot();
updateCharts(); 