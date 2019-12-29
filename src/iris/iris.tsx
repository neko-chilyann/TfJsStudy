import React, { Component } from "react";
import * as tf from '@tensorflow/tfjs';
import * as tfVis from '@tensorflow/tfjs-vis';
import { getIrisData, IRIS_CLASSES } from './data';

/**
 *
 *
 * @export
 * @class iris
 * @extends {Component}
 */
export default class iris extends Component<any, any> {

    /**
     * 训练模型
     *
     * @protected
     * @type {tf.Sequential}
     * @memberof iris
     */
    protected model!: tf.Sequential;

    /**
     * Creates an instance of iris.
     * @param {*} props
     * @memberof iris
     */
    constructor(props: any) {
        super(props);
        this.state = {
            x: '',
            y: '',
            x2: '',
            y2: ''
        };
    }

    /**
     * 组件加载完毕
     *
     * @memberof iris
     */
    public componentDidMount() {
        this.init();
    }

    /**
     * 初始化
     *
     * @protected
     * @returns {Promise<void>}
     * @memberof iris
     */
    protected async init(): Promise<void> {
        // xTrain：训练集的特征   yTrain：训练集的输出  xTest：验证集特征   yTest：验证集特征
        const [xTrain, yTrain, xTest, yTest]: any = getIrisData(0.15);// getIrisData：参数为将数据集的多少分为验证集
        // 初始化模型
        this.model = tf.sequential();
        // 添加层
        this.model.add(tf.layers.dense({
            units: 10,
            inputShape: [xTrain.shape[1]],
            activation: 'sigmoid'
        }));
        this.model.add(tf.layers.dense({
            units: 3,
            activation: 'softmax'
        }));
        // 设置损失函数   设置优化器   设置（metrics）准确度
        this.model.compile({ loss: 'categoricalCrossentropy', optimizer: tf.train.adam(0.1), metrics: ['accuracy'] });
        // 训练
        await this.model.fit(xTrain, yTrain, {
            epochs: 100,
            validationData: [xTest, yTest],
            // loss：训练集损失   val_loss：查看验证集损失   acc：训练集准确度   val_acc：验证集准确度
            callbacks: tfVis.show.fitCallbacks(
                { name: '训练效果' },
                ['loss', 'val_loss', 'acc', 'val_acc'],
                { callbacks: ['onEpochEnd'] }
            )
        });
    }

    /**
     * 预测点击
     *
     * @protected
     * @memberof LogisticRegression
     */
    protected btnClick(): void {
        const output: any = this.model.predict(tf.tensor([[this.state.x * 1, this.state.y * 1, this.state.x2 * 1, this.state.y2 * 1]]));
        alert(`预测结果：${IRIS_CLASSES[output.argMax(1).dataSync(0)]}`);
    }

    /**
     * 值变更
     *
     * @protected
     * @memberof LogisticRegression
     */
    protected change(name: string, e: any): void {
        const data: any = {};
        data[name] = e.target.value;
        this.setState(data);
    }

    /**
     * 绘制内容
     *
     * @returns
     * @memberof iris
     */
    public render() {
        return <div>
            花萼长度：<input type="text" value={this.state.x} onChange={(e: any) => this.change('x', e)} /><br />
            花萼宽度：<input type="text" value={this.state.y} onChange={(e: any) => this.change('y', e)} /><br />
            花瓣长度：<input type="text" value={this.state.x2} onChange={(e: any) => this.change('x2', e)} /><br />
            花瓣宽度：<input type="text" value={this.state.y2} onChange={(e: any) => this.change('y2', e)} /><br />
            <button onClick={() => this.btnClick()}>预测</button>
        </div>;
    }

}