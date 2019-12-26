import React, { Component } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as tfVis from '@tensorflow/tfjs-vis';

/**
 * 线性回归示例
 *
 * @export
 * @class LinerRegression
 * @extends {Component}
 */
export default class LinerRegression extends Component {

    /**
     * 
     *
     * @memberof LinerRegression
     */
    public async componentDidMount() {
        // 训练数据集
        const xs = [1, 2, 3, 4, 5, 6];
        const ys = [4, 8, 12, 16, 20, 24];
        // 绘制散点图
        tfVis.render.scatterplot(
            // 名称
            { name: '线性回归训练集' },
            // 数据
            { values: xs.map((x, i) => ({ x, y: ys[i] })) },
            // 范围
            { xAxisDomain: [0, 7], yAxisDomain: [0, 25] }
        );
        // 初始化模型
        const model = tf.sequential(); // sequential连续的模型
        // dense：全连接层   units：神经元个数   inputShape：形状、维度
        model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
        // 损失函数：均方误差(MSE) 优化器：随机梯度下降(SGD)
        model.compile({ loss: tf.losses.meanSquaredError, optimizer: tf.train.sgd(0.06) });
        // 训练
        const inputs = tf.tensor(xs);
        const labels = tf.tensor(ys);
        await model.fit(inputs, labels, {
            // 小批量
            batchSize: 6,
            // 迭代训练数据量的次数
            epochs: 300,
            // 可视化展示训练过程
            callbacks: tfVis.show.fitCallbacks(
                { name: '训练过程' },
                ['loss']
            )
        });
        // 预测
        const output: any = model.predict(tf.tensor([7, 8, 9, 10]));
        // 预测值
        console.log(output.dataSync());
    }

    /**
     * 
     *
     * @returns
     * @memberof LinerRegression
     */
    public render() {
        return <div className="container"></div>;
    }

}