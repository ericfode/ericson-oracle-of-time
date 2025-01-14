# Time Oracle 🔮

An interactive visualization exploring human timing patterns and predictability. Watch as the oracle learns your rhythm and tries to predict your next move!

## Features
- Real-time visualization of human and random timing patterns
- FFT-based rhythm analysis and prediction
- Interactive state space visualization
- Correlation analysis between human, random, and predicted patterns
- Adjustable prediction lead time

## Live Demo
You can try it out here: [Time Oracle Demo](https://time-oracle.netlify.app)

## Local Development
1. Clone this repository
2. Serve the files using any static file server. For example:
   - Using Python: `python -m http.server 8000`
   - Using Node.js: `npx serve`
   - Using PHP: `php -S localhost:8000`
3. Open `http://localhost:8000` in your browser

## How to Use
1. Press and hold the 'A' key to control the left dot
2. Watch the middle dot try to predict your timing
3. The right dot provides random timing for comparison
4. Adjust the prediction lead time using the slider
5. Watch the various charts to understand the patterns

## Technical Details
- Uses Chart.js for visualizations
- Implements FFT for frequency analysis
- Uses Hanning window for better spectral analysis
- Adaptive prediction based on harmonic analysis

## Deployment
This is a static site that can be deployed anywhere. Recommended platforms:
- Netlify
- GitHub Pages
- Vercel
- Any static file hosting

## License
MIT License - Feel free to use, modify, and distribute! 