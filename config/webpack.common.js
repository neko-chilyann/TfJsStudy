const path = require('path');

module.exports = {
    entry: './src/index.ts',
    output: {
        path: path.resolve('dist'),
        filename: '[name].[hash:8].js',
    },
    module: {
        rules: [{
            test: /\.ts$/,
            use: "ts-loader"
        }]
    },
    resolve: {
        extensions: [
            '.ts'
        ]
    }
};