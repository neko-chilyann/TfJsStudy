import React, { Component } from "react";
import * as tf from '@tensorflow/tfjs';
import * as tfVis from '@tensorflow/tfjs-vis';
import { getData } from './data';

/**
 * 逻辑回归学习
 *
 * @export
 * @class LogisticRegression
 * @extends {Component}
 */
export default class LogisticRegression extends Component<any, any> {

    /**
     * tenserflow模型
     *
     * @protected
     * @memberof LogisticRegression
     */
    protected model!: tf.Sequential;

    /**
     * Creates an instance of LogisticRegression.
     * @param {*} props
     * @memberof LogisticRegression
     */
    constructor(props: any) {
        super(props);
        this.state = {
            x: 0,
            y: 0
        };
    }

    /**
     * 组件挂载完毕
     *
     * @returns {Promise<void>}
     * @memberof LogisticRegression
     */
    public async componentDidMount(): Promise<void> {
        this.init();
    }

    /**
     * 初始化
     *
     * @protected
     * @returns {Promise<void>}
     * @memberof LogisticRegression
     */
    protected async init(): Promise<void> {
        // 生成数据集
        const data = getData(400);
        tfVis.render.scatterplot(
            { name: '逻辑回归学习' },
            {
                values: [
                    data.filter(p => p.label === 1),
                    data.filter(p => p.label === 0)
                ]
            }
        );
        // 转换数据集为tensor
        const inputs = tf.tensor(data.map(p => [p.x, p.y]));
        const labels = tf.tensor(data.map(p => p.label));
        // 初始化模型
        this.model = tf.sequential();
        // 添加神经层 dense：全连接层    units: 神经元个数   inputShape: 输入数据形状   activation：激活函数(sigmoid： 压缩输出值)
        this.model.add(tf.layers.dense({ units: 1, inputShape: [2], activation: 'sigmoid' }));
        // 损失函数(loss) 优化器(optimizer)
        this.model.compile({ loss: tf.losses.logLoss, optimizer: tf.train.adam(0.1) });
        // 训练
        await this.model.fit(inputs, labels, {
            batchSize: 40,
            epochs: 50,
            callbacks: tfVis.show.fitCallbacks(
                { name: '逻辑回归训练过程' },
                ['loss']
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
        const output: any = this.model.predict(tf.tensor([[this.state.x * 1, this.state.y * 1]]));
        alert(output.dataSync());
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
     * @memberof LogisticRegression
     */
    public render() {
        return <div className="logistic-regression">
            <input type="text" value={this.state.x} onChange={(e: any) => this.change('x', e)}/>
            <input type="text" value={this.state.y} onChange={(e: any) => this.change('y', e)}/>
            <button onClick={() => this.btnClick()}>预测</button>
        </div>;
    }

}