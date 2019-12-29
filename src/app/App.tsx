import React, { Component } from 'react';
import LinerRegression from '../liner-regerssion/LinerRegerssion';
import LogisticRegression from '../logistic-regression/logistic-regression';

export default class App extends Component {

    public render() {
        return (
            <div className="App">
                // 线性回归
                {/* <LinerRegression /> */}
                // 逻辑回归
                <LogisticRegression />
            </div>
        );
    }

}
